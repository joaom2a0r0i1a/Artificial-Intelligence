# Artificial-Intelligence
Project for the Artificial Intelligence course from IST during the summer semester of 2023. 

The project involves the use of search algorithms in order to solve the Bimaru game. 

Bimaru is a puzzle inspired by the battleship game between two players. It consists of a standard 10x10 board representing an ocean area that has a hidden fleet of one four-square-length boat, two three-square-length boats, three two-square-length boats, and four one-square-length boats. These boats could be arranged either vertically or horizontally. Furthermore, the player is also given the number of occupied squares in the line and column and various tips. Each tip specifies the state of a square on the board: water (empty square); circle (one-square-length boat); middle (of either a four-square or three-square-length boat); up, down, left, and right. In this project, the tips will be represented as uppercase, and the rest of the guessed positions will be represented as lowercase (except for water which will be represented as a '.'). Intuitively, the states will be represented as 'W' or '.' for water, 'c' or 'C' for circle, 'm' or 'M' for middle, 'u' or 'U' for up, 'd' or 'D' for down, 'l' or 'L' for left, and 'r' or 'R' for right. 

The instances given represent cases where the tips and number of occupied squares in the lines and columns differ. Each case provides different space and time complexities to solve the game. The objective is to solve the puzzle as fast as possible.

The tips and number of occupied squares in lines and columns are in the txt files. In order to run the code, the following line should be used in the prompt:

```
python3 bimaru.py < <instance_file>
```
