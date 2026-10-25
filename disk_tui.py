import os
import sys
import curses
import shutil
import time

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:6.1f} {unit}"
        size /= 1024.0
    return f"{size:6.1f} PB"

class Node:
    def __init__(self, name, parent, is_dir):
        self.name = name
        self.parent = parent
        self.is_dir = is_dir
        self.children = []
        self.size = 0
        self.count = 0
        self.error = False

    def get_path(self):
        if self.parent is None:
            return self.name
        return os.path.join(self.parent.get_path(), self.name)

def build_tree(root_path, stdscr):
    root = Node(root_path, None, True)
    
    # Non-recursive walk using a stack to avoid recursion limit and update UI
    stack = [root]
    files_scanned = 0
    last_update = time.time()
    
    while stack:
        curr = stack.pop()
        path = curr.get_path()
        
        try:
            with os.scandir(path) as it:
                for entry in it:
                    files_scanned += 1
                    if files_scanned % 500 == 0:
                        now = time.time()
                        if now - last_update > 0.1:
                            h, w = stdscr.getmaxyx()
                            stdscr.clear()
                            stdscr.addstr(h//2, max(0, w//2 - 15), f"Scanning... {files_scanned} items found")
                            stdscr.addstr(h//2+1, max(0, w//2 - min(len(path), w)//2), path[-w+2:])
                            stdscr.refresh()
                            last_update = now
                            
                    if entry.is_symlink():
                        continue
                    if entry.is_file(follow_symlinks=False):
                        size = entry.stat(follow_symlinks=False).st_size
                        child = Node(entry.name, curr, False)
                        child.size = size
                        child.count = 1
                        curr.children.append(child)
                        
                        # update parents
                        p = curr
                        while p:
                            p.size += size
                            p.count += 1
                            p = p.parent
                    elif entry.is_dir(follow_symlinks=False):
                        child = Node(entry.name, curr, True)
                        curr.children.append(child)
                        stack.append(child)
        except PermissionError:
            curr.error = True
        except OSError:
            curr.error = True
            
    # Sort children by size descending
    def sort_tree(node):
        if node.is_dir:
            node.children.sort(key=lambda x: x.size, reverse=True)
            for c in node.children:
                sort_tree(c)
    sort_tree(root)
    return root

def main(stdscr):
    curses.curs_set(0)
    curses.use_default_colors()
    
    try:
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Highlight
        curses.init_pair(2, curses.COLOR_CYAN, -1)                   # Directories
        curses.init_pair(3, curses.COLOR_RED, -1)                    # Errors
    except Exception:
        pass

    target_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    
    root = build_tree(target_dir, stdscr)
    curr_node = root
    selected_idx = 0
    scroll_offset = 0

    while True:
        h, w = stdscr.getmaxyx()
        stdscr.clear()

        # Gather disk info
        try:
            total, used, free = shutil.disk_usage(curr_node.get_path())
        except Exception:
            total, used, free = 0, 0, 0
            
        header1 = f" Path: {curr_node.get_path()} "
        header2 = f" Disk Total: {format_size(total)} | Used: {format_size(used)} | Free: {format_size(free)} "
        stdscr.addnstr(0, 0, header1.ljust(w), w, curses.A_REVERSE)
        stdscr.addnstr(1, 0, header2.ljust(w), w, curses.A_REVERSE)
        
        col_name = "Name".ljust(max(10, w - 25))
        col_size = "Size".rjust(10)
        col_items = "Items".rjust(10)
        stdscr.addnstr(2, 0, f"  {col_name}{col_size}{col_items}", w, curses.A_BOLD)
        
        list_h = h - 5
        
        if len(curr_node.children) == 0:
            stdscr.addstr(4, 2, "Directory is empty or inaccessible.")
        else:
            for i in range(list_h):
                idx = scroll_offset + i
                if idx >= len(curr_node.children):
                    break
                    
                child = curr_node.children[idx]
                is_selected = (idx == selected_idx)
                
                attr = curses.color_pair(1) if is_selected else curses.A_NORMAL
                name_attr = curses.color_pair(2) | curses.A_BOLD if child.is_dir else curses.A_NORMAL
                if is_selected:
                    name_attr = attr
                
                prefix = "> " if is_selected else "  "
                name_disp = (child.name + "/") if child.is_dir else child.name
                if child.error:
                    name_disp += " [ERROR]"
                    name_attr = curses.color_pair(3)
                
                # Truncate name to fit
                name_w = max(10, w - 25)
                if len(name_disp) > name_w:
                    name_disp = name_disp[:name_w-3] + "..."
                else:
                    name_disp = name_disp.ljust(name_w)
                    
                size_str = format_size(child.size).rjust(10)
                count_str = str(child.count).rjust(10) if child.is_dir else "".rjust(10)
                
                # Draw
                stdscr.addstr(3 + i, 0, prefix, attr)
                stdscr.addstr(3 + i, 2, name_disp, name_attr)
                stdscr.addstr(3 + i, 2 + name_w, size_str, attr)
                stdscr.addstr(3 + i, 2 + name_w + 10, count_str, attr)

        footer = " UP/DOWN: Move | RIGHT/ENTER: Enter | LEFT/BACKSPACE: Up | Q: Quit "
        try:
            # ljust to w-1 to avoid writing to the very bottom-right cell, which causes a curses ERR
            stdscr.addnstr(h - 1, 0, footer.ljust(w - 1), w - 1, curses.A_REVERSE)
        except curses.error:
            pass
        
        stdscr.refresh()
        
        c = stdscr.getch()
        if c in [ord('q'), ord('Q')]:
            break
        elif c == curses.KEY_UP:
            if selected_idx > 0:
                selected_idx -= 1
                if selected_idx < scroll_offset:
                    scroll_offset = selected_idx
        elif c == curses.KEY_DOWN:
            if selected_idx < len(curr_node.children) - 1:
                selected_idx += 1
                if selected_idx >= scroll_offset + list_h:
                    scroll_offset = selected_idx - list_h + 1
        elif c in [curses.KEY_RIGHT, 10, 13]: # Enter folder
            if len(curr_node.children) > 0 and curr_node.children[selected_idx].is_dir:
                curr_node = curr_node.children[selected_idx]
                selected_idx = 0
                scroll_offset = 0
        elif c in [curses.KEY_LEFT, curses.KEY_BACKSPACE, 127, 8]: # Go up
            if curr_node.parent is not None:
                # find where curr_node is in parent's children to restore selection
                selected_idx = curr_node.parent.children.index(curr_node)
                curr_node = curr_node.parent
                scroll_offset = max(0, selected_idx - list_h // 2)

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
