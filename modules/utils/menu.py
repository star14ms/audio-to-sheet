from simple_term_menu import TerminalMenu
import os


def main():
    options = ["entry 1", "entry 2", "entry 3"]
    terminal_menu = TerminalMenu(options)
    menu_entry_index = terminal_menu.show()
    print(f"You have selected {options[menu_entry_index]}!")


def show_menu(menu_ops: dict[dict[str: str]], title="Select", quit_choice="[q] 나가기", verbose=True):
    """
    Example
    ========
        >>> value, key = show_menu({'[a] Choice-1' : 1, '[b] Choice-2': 2, '[q] Quit': None}, title="Select")
        # (a, b, q - keyboard shortcuts)
    """

    mainMenu = TerminalMenu(list(menu_ops.keys()), title=title)

    exit = False
    while not exit:
        optionsIndex = mainMenu.show()
        choice = list(menu_ops.keys())[optionsIndex]
    
        if (choice == quit_choice):
            exit = True
        else:
            if verbose: print(choice + "\n")

            if isinstance(menu_ops[choice], dict):
                value, _ = show_menu(menu_ops[choice], title=choice)
                if value is not None:
                    return value, choice
                    
            else:
                return menu_ops[choice], " ".join(choice.split()[1:])
    
    return None, None


# Already using shortcut keys: k⬅️, h⬆️, m➡️, p⬇️, q(나가기) # ←→↑↓
shortcut_keys = '1234567890abcdefgijlnorstuvwxyz,.;-='


def show_menu_list_dir(view_path):
    options_origin = os.listdir(view_path)
    options = [
        (f'[{shortcut_keys[index]}] ' if index < len(shortcut_keys) else '') + \
        (option + '/' if os.path.isdir(os.path.join(view_path, option)) else option) \
            for index, option in enumerate(options_origin)
    ]
    options.append('[q] ../')

    menu = TerminalMenu(options, title=view_path)
    choice = menu.show()
    choice = options[choice].split('] ')[-1]
    
    if choice == '../':
        if len(view_path.split('/')) > 2:
            view_path = '/'.join(view_path.split('/')[:-2]) + '/'
        else:
            view_path = os.path.abspath(view_path + '/..')
    else:
        view_path = os.path.join(view_path, choice)

    return choice, view_path


def select_file(view_path='./'):
    exit = False

    while not exit:
        choice, view_path = show_menu_list_dir(view_path)
    
        if choice != '/..' and choice[-1] != '/':
            exit = True
    print()
    return os.path.abspath(view_path)



if __name__ == "__main__":
    # main()
    file_path = select_file()
    print(file_path)

    