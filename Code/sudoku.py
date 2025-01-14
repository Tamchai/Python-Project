import cv2
import numpy as np

grid_size = 9
cell_size = 50
board_size = grid_size * cell_size
line_color = (0, 0, 0)
text_color = (0, 0, 0)
input_color = (255, 0, 0)
invalid_color = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ด่านเกม sudoku 
sudoku_grids = [
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ],
    [
        [0, 0, 0, 0, 0, 0, 0, 8, 0],
        [0, 0, 0, 0, 0, 5, 6, 4, 0],
        [0, 1, 0, 6, 0, 0, 3, 0, 9],
        [0, 3, 0, 2, 6, 7, 0, 0, 0],
        [0, 5, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 9, 1, 0, 0, 3],
        [4, 2, 0, 0, 0, 0, 0, 1, 0],
        [7, 0, 9, 0, 0, 0, 4, 6, 0],
        [0, 0, 0, 0, 8, 2, 0, 0, 0]
    ],
    [
        [5, 0, 9, 0, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 6, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 2, 5, 0, 6],
        [0, 0, 8, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 7, 3, 0, 8, 0, 1],
        [2, 0, 1, 6, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 7, 0, 0, 4, 5, 1, 0, 8],
        [4, 0, 2, 1, 9, 0, 0, 0, 0],
    ],
    [
        [0, 0, 8, 0, 0, 0, 0, 0, 0],
        [9, 0, 0, 0, 4, 6, 5, 0, 7],
        [0, 4, 0, 0, 0, 5, 2, 0, 0],
        [2, 0, 9, 0, 8, 4, 0, 3, 0],
        [0, 6, 0, 0, 9, 0, 0, 0, 0],
        [4, 0, 0, 6, 2, 0, 7, 0, 1],
        [0, 0, 0, 7, 0, 0, 0, 8, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 6],
        [0, 0, 0, 0, 0, 0, 0, 7, 0],
    ],
    [
        [0, 0, 8, 3, 0, 0, 0, 0, 0],
        [0, 0, 5, 4, 0, 0, 0, 3, 0],
        [0, 6, 1, 0, 0, 8, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 9, 7, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 8, 0, 0, 7, 0, 4],
        [0, 2, 0, 0, 0, 9, 0, 0, 1],
        [0, 5, 0, 7, 0, 3, 8, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 2, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 2, 0, 7],
        [0, 0, 0, 8, 5, 0, 0, 0, 6],
        [0, 0, 0, 2, 0, 0, 1, 0, 0],
        [0, 0, 2, 1, 0, 0, 9, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 6, 0, 8, 3],
        [0, 7, 0, 0, 0, 0, 0, 0, 0],
        [6, 0, 0, 5, 0, 9, 0, 0, 0],
        [8, 9, 0, 0, 0, 3, 0, 6, 0],
    ],
    [
        [2, 0, 0, 0, 0, 0, 0, 5, 0],
        [0, 0, 3, 0, 2, 0, 4, 0, 0],
        [0, 0, 6, 0, 0, 1, 9, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 8, 0],
        [0, 0, 0, 6, 0, 8, 0, 0, 0],
        [0, 4, 9, 0, 0, 5, 0, 0, 0],
        [6, 9, 0, 0, 1, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 8, 0, 0, 0, 7, 0, 0, 5],
    ],
    [
        [0, 0, 0, 0, 9, 0, 0, 0, 0],
        [5, 0, 0, 0, 8, 0, 0, 9, 0],
        [0, 0, 3, 6, 0, 7, 0, 0, 0],
        [7, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 6, 8, 3, 9],
        [0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 8, 0, 0, 0, 1, 0, 6],
        [0, 2, 0, 0, 0, 4, 0, 0, 0],
        [0, 3, 6, 7, 0, 0, 0, 0, 0],
    ],
    [
        [0, 4, 0, 0, 0, 0, 3, 5, 0],
        [0, 0, 0, 2, 0, 4, 0, 0, 0],
        [0, 0, 3, 0, 5, 0, 0, 0, 7],
        [9, 0, 0, 0, 0, 8, 1, 0, 6],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 0, 0, 1, 0, 0, 0],
        [3, 0, 0, 0, 7, 0, 4, 0, 2],
        [7, 0, 0, 0, 0, 9, 0, 0, 8],
        [0, 0, 2, 0, 0, 0, 0, 0, 0],
    ],
    [
        [1, 7, 0, 0, 0, 6, 0, 0, 0],
        [0, 0, 0, 0, 7, 3, 0, 2, 6],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 1, 7, 0],
        [0, 0, 0, 5, 1, 0, 0, 0, 0],
        [3, 6, 0, 0, 0, 0, 0, 4, 0],
        [0, 9, 0, 0, 3, 0, 4, 0, 0],
        [0, 0, 0, 8, 0, 0, 5, 0, 2],
        [7, 3, 0, 0, 0, 0, 0, 0, 0],
    ]
]

level_index = 0
sudoku_grid = sudoku_grids[level_index]
default_cells = {(row, col) for row in range(grid_size) for col in range(grid_size) if sudoku_grid[row][col] != 0}
input_cells = {}
invalid_cells = set()
incorrect_input_count = 0
selected_cell = None

def is_valid_move(grid, row, col, num): # เช็คค่าแนวตั้ง แนวนอน ตาราง 3*3
    for i in range(grid_size):
        if grid[row][i] == num or grid[i][col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if grid[start_row + i][start_col + j] == num:
                return False
    return True

def draw_grid(img, grid, input_cells, invalid_cells, selected_cell): # สร้างตาราง
    for i in range(grid_size + 1):
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(img, (0, i * cell_size), (board_size, i * cell_size), line_color, thickness)
        cv2.line(img, (i * cell_size, 0), (i * cell_size, board_size), line_color, thickness)

    if selected_cell is not None:
        sel_row, sel_col = selected_cell
        cv2.rectangle(img, (sel_col * cell_size, sel_row * cell_size), 
                      ((sel_col + 1) * cell_size, (sel_row + 1) * cell_size), 
                      (255, 26, 0), 2)

    for row in range(grid_size):
        for col in range(grid_size):
            if grid[row][col] != 0:
                text = str(grid[row][col])
                x = col * cell_size + cell_size // 4
                y = row * cell_size + int(cell_size * 0.75)
                color = text_color if (row, col) in default_cells else input_color
                cv2.putText(img, text, (x, y), FONT, 1, color, 2)

    for (row, col), value in input_cells.items():
        x = col * cell_size + cell_size // 4
        y = row * cell_size + int(cell_size * 0.75)
        color = invalid_color if (row, col) in invalid_cells else input_color
        cv2.putText(img, str(value), (x, y), FONT, 1, color, 2)

def mouse_click(event, x, y, flags, param): # รับค่าจากเมาส์
    global selected_cell
    if event == cv2.EVENT_LBUTTONDOWN:
        row = y // cell_size
        col = x // cell_size
        if (row, col) not in default_cells:
            selected_cell = (row, col)

def check_win(grid, input_cells): #เช็คชนะ
    for row in range(grid_size):
        for col in range(grid_size):
            if (row, col) not in default_cells:
                if (row, col) not in input_cells or not is_valid_move(grid, row, col, input_cells[(row, col)]):
                    return False
    return True

def start_new_level(): #เปลี่ยนด่าน
    global level_index, sudoku_grid, default_cells, input_cells, invalid_cells, incorrect_input_count, selected_cell
    level_index += 1
    if level_index < len(sudoku_grids):
        sudoku_grid = sudoku_grids[level_index]
        default_cells = {(row, col) for row in range(grid_size) for col in range(grid_size) if sudoku_grid[row][col] != 0}
        input_cells = {}
        invalid_cells = set()
        incorrect_input_count = 0
        selected_cell = None

        img[:] = 255
        draw_grid(img, sudoku_grid, input_cells, invalid_cells, selected_cell)
        cv2.imshow("Sudoku", img)

img = np.ones((board_size, board_size, 3), dtype=np.uint8) * 255
draw_grid(img, sudoku_grid, input_cells, invalid_cells, None)

cv2.imshow("Sudoku", img)
cv2.setMouseCallback("Sudoku", mouse_click)

while True: # เริ่มเกม
    cv2.imshow("Sudoku", img)
    key = cv2.waitKey(1)

    if key in [ord(str(i)) for i in range(1, 10)] and selected_cell:
        row, col = selected_cell
        num = int(chr(key))

        if is_valid_move(sudoku_grid, row, col, num):
            input_cells[(row, col)] = num
            invalid_cells.discard((row, col))
        else:
            if (row, col) not in invalid_cells:
                incorrect_input_count += 1
            invalid_cells.add((row, col))
            input_cells[(row, col)] = num

        img[:] = 255
        draw_grid(img, sudoku_grid, input_cells, invalid_cells, selected_cell)
        selected_cell = None

        if check_win(sudoku_grid, input_cells):
            cv2.rectangle(img, (0, board_size // 2 - 50),
                          (board_size, board_size // 2 + 100), (200, 200, 200), -1)

            win_text = "You win!!"
            incorrect_text = f"Incorrect inputs: {incorrect_input_count}"

            win_text_size = cv2.getTextSize(win_text, FONT, 2, 3)[0]
            incorrect_text_size = cv2.getTextSize(incorrect_text, FONT, 1, 2)[0]

            win_text_x = (board_size - win_text_size[0]) // 2
            win_text_y = board_size // 2

            incorrect_text_x = (board_size - incorrect_text_size[0]) // 2
            incorrect_text_y = win_text_y + 50

            cv2.putText(img, win_text, (win_text_x, win_text_y), FONT, 2, (0, 255, 0), 3)
            cv2.putText(img, incorrect_text, (incorrect_text_x, incorrect_text_y), FONT, 1, (0, 0, 255), 2)
            cv2.imshow("Sudoku", img)
            cv2.waitKey(10000)
            start_new_level()

    if key == 8 and selected_cell:
        row, col = selected_cell
        if (row, col) in input_cells:
            del input_cells[(row, col)]
        invalid_cells.discard((row, col))

        img[:] = 255
        draw_grid(img, sudoku_grid, input_cells, invalid_cells, selected_cell)
        selected_cell = None

    if key == 27:  # ESC key to exit
        break
    
    if cv2.getWindowProperty("Sudoku",cv2.WND_PROP_VISIBLE) < 1:
        break
    
cv2.destroyAllWindows()