# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 02:
# 95804 João Mendes
# 95807 João Peixoto

import numpy as np
import copy
import sys

from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class BimaruState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self, rows, cols):
        """Inicializa variáveis básicas: tabuleiro, tamanho do tabuleiro, 
        número de barcos esperados nas linhas e colunas, número de barcos 
        com um certo tamanho, o atual número de barcos nas linhas e 
        colunas, as dicas e o número de células com água"""
        self.boats_rows = np.array([int(x) for x in rows])
        self.boats_columns = np.array([int(x) for x in cols])
        self.size_rows = len(cols)
        self.size_columns = len(rows)
        self.board = np.full((self.size_rows, self.size_columns), None)
        self.size_4 = 0
        self.size_3 = 0
        self.size_2 = 0
        self.size_1 = 0
        self.current_boats_rows = np.zeros(self.size_rows)
        self.current_boats_columns = np.zeros(self.size_columns)
        self.hints = []
        
    def set_value(self, row: int, col: int, value: str):
        """Define novo valor na posição especificada do tabuleiro."""
        self.board[row, col] = value

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row, col]

    def count_boat_sizes(self):
        """Calcula o tamanho dos barcos existentes no tabuleiro 
        e o número de barcos existentes em cada linha e coluna do tabuleiro."""

        # Inicializa as variáveis a calcular
        self.size_4 = 0
        self.size_3 = 0
        self.size_2 = 0
        self.size_1 = 0
        self.current_boats_rows = np.zeros(self.size_rows)
        self.current_boats_columns = np.zeros(self.size_columns)

        current_boats = np.zeros((self.size_rows, self.size_columns))
        
        # Calcula o tamanho dos barcos
        for row in range(self.size_rows):
            for col in range(self.size_columns):
                if row < self.size_rows-3 and self.get_value(row, col) in ['t', 'T'] and self.get_value(row+1, col) in ['m', 'M'] and self.get_value(row+2, col) in ['m', 'M'] and self.get_value(row+3, col) in ['b', 'B']:
                    self.size_4 += 1
                    current_boats[row:row+4, col] += 1
                if col < self.size_columns-3 and self.get_value(row, col) in ['l', 'L'] and self.get_value(row, col+1) in ['m', 'M'] and self.get_value(row, col+2) in ['m', 'M'] and self.get_value(row, col+3) in ['r', 'R']:
                    self.size_4 += 1
                    current_boats[row, col:col+4] += 1
                if row < self.size_rows-2 and self.get_value(row, col) in ['t', 'T'] and self.get_value(row+1, col) in ['m', 'M'] and self.get_value(row+2, col) in ['b', 'B']:
                    self.size_3 += 1
                    current_boats[row:row+3, col] += 1
                if col < self.size_columns-2 and self.get_value(row, col) in ['l', 'L'] and self.get_value(row, col+1) in ['m', 'M'] and self.get_value(row, col+2) in ['r', 'R']:
                    self.size_3 += 1
                    current_boats[row, col:col+3] += 1
                if row < self.size_rows-1 and self.get_value(row, col) in ['t', 'T'] and self.get_value(row+1, col) in ['b', 'B']:
                    self.size_2 += 1
                    current_boats[row:row+2, col] += 1
                if col < self.size_columns-1 and self.get_value(row, col) in ['l', 'L'] and self.get_value(row, col+1) in ['r', 'R']:
                    self.size_2 += 1
                    current_boats[row, col:col+2] += 1
                if self.get_value(row, col) in ['c', 'C']:
                    self.size_1 += 1
                    current_boats[row, col] += 1

        self.current_boats_rows = np.sum(current_boats, axis=1)
        self.current_boats_columns = np.sum(current_boats, axis=0)

    def fill_full_rows(self):
        """Coloca água na linha se o número de barcos na linha 
        corresponder ao número de barcos esperados na mesma."""
        boats_row_counter = np.zeros(self.size_rows)
        for row in range(self.size_rows):
            empty_columns = np.array([], dtype=np.int32)

            for aux_col in range(self.size_columns):
                if self.board[row, aux_col] not in [None, 'W', '.']:
                    boats_row_counter[row] += 1
                elif self.board[row, aux_col] is None:
                    empty_columns = np.append(empty_columns, aux_col)

            if boats_row_counter[row] == self.boats_rows[row]:
                for i in empty_columns:
                    self.board[row, i] = '.'
    
    def fill_full_columns(self):
        """Coloca água na coluna se o número de barcos na coluna 
        corresponder ao número de barcos esperados na mesma."""
        boats_column_counter = np.zeros(self.size_columns)
        for col in range(self.size_columns):
            empty_rows = np.array([], dtype=np.int32)

            for aux_row in range(self.size_rows):
                if self.board[aux_row, col] not in [None, 'W', '.']:
                    boats_column_counter[col] += 1
                elif self.board[aux_row, col] is None:
                    empty_rows = np.append(empty_rows, aux_row)

            if boats_column_counter[col] == self.boats_columns[col]:
                for i in empty_rows:
                    self.board[i, col] = '.'

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        below = None
        above = None

        if row > 0:
            below = self.board[row-1, col]
        if row < 9:
            above = self.board[row+1, col]

        return (above, below)


    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        left = None
        right = None

        if col > 0:
            left = self.board[row, col-1]
        if col < 9:
            right = self.board[row, col+1]

        return (left, right)

    def print(self):
        """Imprime o tabuleiro."""
        for row in self.board:
            print("".join(row))


    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 bimaru.py < input_T01

            > from sys import stdin
            > line = stdin.readline().split()
        """
        # TODO

        # Inicializa as dicas e o tabuleiro
        rows = []
        cols = []
        hints = []

        # Lê as linhas do input
        lines = sys.stdin.readlines()

        # Lê o tabuleiro e as dicas do tabuleiro
        for line in lines:
            line = line.split()
            if line[0] == 'ROW':
                rows.extend(line[1:])
            elif line[0] == 'COLUMN':
                cols.extend(line[1:])
            elif line[0].isdigit():
                num_hints = int(line[0])
            elif line[0] == 'HINT':
                hints.append((int(line[1]), int(line[2]), line[3]))
            else:
                break
        
        # Cria o objeto Board inicial
        board_init = Board(rows, cols)

        for hint in hints:
            row, col, value = hint
            board_init.set_value(row, col, value)
            board_init.hints.append(hint)

        board_init.fill_full_rows()
        board_init.fill_full_columns()
        board_init.count_boat_sizes()

        return board_init


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = BimaruState(board)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # lista de ações possíveis
        actions = []

        # lista auxiliar
        actions_aux = []

        # Percorre todo o tabuleiro e procura sítios válidos onde colocar barcos e as águas à sua volta
        for row in range(state.board.size_rows):
            for col in range(state.board.size_columns):
                # Coloca o barco de 4 peças em primeiro lugar
                if state.board.size_4 < 1:      
                    # 4 peças na vertical - verifica se há espaço suficiente na coluna e se as células estão vazias ou com uma dica válida
                    if row < state.board.size_rows-3 and state.board.get_value(row, col) in [None, 'T'] and state.board.get_value(row+1, col) in [None, 'M'] and state.board.get_value(row+2, col) in [None, 'M'] and state.board.get_value(row+3, col) in [None, 'B'] and state.board.boats_columns[col] - state.board.current_boats_columns[col] >= 4:
                        # Lista de células adjacentes
                        neighbors = [(row-1, col), (row-1, col-1), (row-1, col+1), (row+4, col), (row+4, col-1), (row+4, col+1), (row, col-1), (row+1, col-1), (row+2, col-1), (row+3, col-1), (row, col+1), (row+1, col+1), (row+2, col+1), (row+3, col+1)]
                        pieces_surrounding = False
                        # Verifica se alguma célula adjacente já tem peças
                        for r, c in neighbors:
                            if r < 0 or r > state.board.size_rows-1 or c < 0 or c > state.board.size_columns-1:
                                continue  # Ignorar células fora do tabuleiro
                                
                            value = state.board.get_value(r, c)
                            if value not in [None, 'W', '.']:
                                pieces_surrounding = True
                                break

                        # Coloca o barco se não houver peças adjacentes  
                        # Não verifica se está a escrever por cima de uma dica, as dicas são colocadas novamente após encontrar a solução
                        if not pieces_surrounding:
                            actions_aux.extend([(row, col, 't'), (row+1, col, 'm'), (row+2, col, 'm'), (row+3, col, 'b')])      # barco

                            # Coloca as águas à volta do barco em posições válidas
                            if row > 0:
                                actions_aux.append((row-1, col, '.'))
                                if col > 0:
                                    actions_aux.append((row-1, col-1, '.'))
                                if col < 9: 
                                    actions_aux.append((row-1, col+1, '.'))

                            if row+3 < 9:
                                actions_aux.append((row+4, col, '.'))
                                if col > 0:
                                    actions_aux.append((row+4, col-1, '.'))
                                if col < 9: 
                                    actions_aux.append((row+4, col+1, '.'))

                            if col > 0:
                                actions_aux.extend([(row, col-1, '.'), (row+1, col-1, '.'), (row+2, col-1, '.'), (row+3, col-1, '.')])

                            if col < 9:
                                actions_aux.extend([(row, col+1, '.'), (row+1, col+1, '.'), (row+2, col+1, '.'), (row+3, col+1, '.')])

                        # Adicionar todas as peças a colocar à lista de ações
                        if actions_aux:
                            actions.append(actions_aux)
                            actions_aux = []

                    # 4 peças na horizontal - verifica se há espaço suficiente na linha e se as células estão vazias ou com uma dica válida
                    if col < state.board.size_columns-3 and state.board.get_value(row, col) in [None, 'L'] and state.board.get_value(row, col+1) in [None, 'M'] and state.board.get_value(row, col+2) in [None, 'M'] and state.board.get_value(row, col+3) in [None, 'R'] and state.board.boats_rows[row] - state.board.current_boats_rows[row] >= 4:
                        # Lista de células adjacentes
                        neighbors = [(row, col-1), (row-1, col-1), (row+1, col-1), (row, col+4), (row-1, col+4), (row+1, col+4), (row-1, col), (row-1, col+1), (row-1, col+2), (row-1, col+3), (row+1, col), (row+1, col+1), (row+1, col+2), (row+1, col+3)]
                        pieces_surrounding = False
                        # Verifica se alguma célula adjacente já tem peças
                        for r, c in neighbors:
                            if r < 0 or r > state.board.size_rows-1 or c < 0 or c > state.board.size_columns-1:
                                continue  # Ignorar células fora do tabuleiro
                                
                            value = state.board.get_value(r, c)
                            if value not in [None, 'W', '.']:
                                pieces_surrounding = True
                                break

                        # Coloca o barco se não houver peças adjacentes 
                        # Não verifica se está a escrever por cima de uma dica, as dicas são colocadas novamente após encontrar a solução  
                        if not pieces_surrounding:
                            actions_aux.extend([(row, col, 'l'), (row, col+1, 'm'), (row, col+2, 'm'), (row, col+3, 'r')])      # barco

                            # Coloca as águas à volta do barco em posições válidas
                            if col > 0:
                                actions_aux.append((row, col-1, '.'))
                                if row > 0:
                                    actions_aux.append((row-1, col-1, '.'))
                                if row < 9:
                                    actions_aux.append((row+1, col-1, '.'))

                            if col+3 < 9:
                                actions_aux.append((row, col+4, '.'))
                                if row > 0:
                                    actions_aux.append((row-1, col+4, '.'))
                                if row < 9:
                                    actions_aux.append((row+1, col+4, '.'))

                            if row > 0:
                                actions_aux.extend([(row-1, col, '.'), (row-1, col+1, '.'), (row-1, col+2, '.'), (row-1, col+3, '.')])

                            if row < 9:
                                actions_aux.extend([(row+1, col, '.'), (row+1, col+1, '.'), (row+1, col+2, '.'), (row+1, col+3, '.')])


                        # Adicionar todas as peças a colocar à lista de ações
                        if actions_aux:
                            actions.append(actions_aux)
                            actions_aux = []

                # Coloca os barcos de 3 peças depois de colocar o barco de 4 peças
                if state.board.size_4 == 1 and state.board.size_3 < 2:
                    # 3 peças na vertical - verifica se há espaço suficiente na coluna e se as células estão vazias ou com uma dica válida
                    if row < state.board.size_rows-2 and state.board.get_value(row, col) in [None, 'T'] and state.board.get_value(row+1, col) in [None, 'M'] and state.board.get_value(row+2, col) in [None, 'B'] and state.board.boats_columns[col] - state.board.current_boats_columns[col] >= 3:
                        # Lista de células adjacentes
                        neighbors = [(row-1, col), (row-1, col-1), (row-1, col+1), (row+3, col), (row+3, col-1), (row+3, col+1), (row, col-1), (row+1, col-1), (row+2, col-1), (row, col+1), (row+1, col+1), (row+2, col+1)]
                        pieces_surrounding = False
                        # Verifica se alguma célula adjacente já tem peças
                        for r, c in neighbors:
                            if r < 0 or r > state.board.size_rows-1 or c < 0 or c > state.board.size_columns-1:
                                continue  # Ignorar células fora do tabuleiro
                                
                            value = state.board.get_value(r, c)
                            if value not in [None, 'W', '.']:
                                pieces_surrounding = True
                                break
                        
                        # Coloca o barco se não houver peças adjacentes
                        # Não verifica se está a escrever por cima de uma dica, as dicas são colocadas novamente após encontrar a solução  
                        if not pieces_surrounding:
                            actions_aux.extend([(row, col, 't'), (row+1, col, 'm'), (row+2, col, 'b')])     # barco

                            # Coloca as águas à volta do barco em posições válidas
                            if row > 0:
                                actions_aux.append((row-1, col, '.'))
                                if col > 0:
                                    actions_aux.append((row-1, col-1, '.'))
                                if col < 9: 
                                    actions_aux.append((row-1, col+1, '.'))

                            if row+2 < 9:
                                actions_aux.append((row+3, col, '.'))
                                if col > 0:
                                    actions_aux.append((row+3, col-1, '.'))
                                if col < 9: 
                                    actions_aux.append((row+3, col+1, '.'))

                            if col > 0:
                                actions_aux.extend([(row, col-1, '.'), (row+1, col-1, '.'), (row+2, col-1, '.')])

                            if col < 9:
                                actions_aux.extend([(row, col+1, '.'), (row+1, col+1, '.'), (row+2, col+1, '.')])

                        # Adicionar todas as peças a colocar à lista de ações
                        if actions_aux:
                            actions.append(actions_aux)
                            actions_aux = []

                    # 3 peças na horizontal - verifica se há espaço suficiente na linha e se as células estão vazias ou com uma dica válida
                    if col < state.board.size_columns-2 and state.board.get_value(row, col) in [None, 'L'] and state.board.get_value(row, col+1) in [None, 'M'] and state.board.get_value(row, col+2) in [None, 'R'] and state.board.boats_rows[row] - state.board.current_boats_rows[row] >= 3:
                        # Lista de células adjacentes
                        neighbors = [(row, col-1), (row-1, col-1), (row+1, col-1), (row, col+3), (row-1, col+3), (row+1, col+3), (row-1, col), (row-1, col+1), (row-1, col+2), (row+1, col), (row+1, col+1), (row+1, col+2)]
                        pieces_surrounding = False
                        # Verifica se alguma célula adjacente já tem peças
                        for r, c in neighbors:
                            if r < 0 or r > state.board.size_rows-1 or c < 0 or c > state.board.size_columns-1:
                                continue  # Ignorar células fora do tabuleiro
                                
                            value = state.board.get_value(r, c)
                            if value not in [None, 'W', '.']:
                                pieces_surrounding = True
                                break
                        
                        # Coloca o barco se não houver peças adjacentes  
                        # Não verifica se está a escrever por cima de uma dica, as dicas são colocadas novamente após encontrar a solução 
                        if not pieces_surrounding:
                            actions_aux.extend([(row, col, 'l'), (row, col+1, 'm'), (row, col+2, 'r')])     # barco

                            # Coloca as águas à volta do barco em posições válidas
                            if col > 0:
                                actions_aux.append((row, col-1, '.'))
                                if row > 0:
                                    actions_aux.append((row-1, col-1, '.'))
                                if row < 9:
                                    actions_aux.append((row+1, col-1, '.'))

                            if col+2 < 9: 
                                actions_aux.append((row, col+3, '.'))
                                if row > 0:
                                    actions_aux.append((row-1, col+3, '.'))
                                if row < 9:
                                    actions_aux.append((row+1, col+3, '.'))

                            if row > 0:
                                actions_aux.extend([(row-1, col, '.'), (row-1, col+1, '.'), (row-1, col+2, '.')])

                            if row < 9:
                                actions_aux.extend([(row+1, col, '.'), (row+1, col+1, '.'), (row+1, col+2, '.')])

                        # Adicionar todas as peças a colocar à lista de ações
                        if actions_aux:
                            actions.append(actions_aux)
                            actions_aux = []

                # Coloca os barcos de 2 peças depois de colocar os barcos de 4 e 3 peças
                if state.board.size_4 == 1 and state.board.size_3 == 2 and state.board.size_2 < 3:
                    # 2 peças na vertical - verifica se há espaço suficiente na coluna e se as células estão vazias ou com uma dica válida
                    if row < state.board.size_rows-1 and state.board.get_value(row, col) in [None, 'T'] and state.board.get_value(row+1, col) in [None, 'B'] and state.board.boats_columns[col] - state.board.current_boats_columns[col] >= 2:
                        # Lista de células adjacentes
                        neighbors = [(row-1, col), (row-1, col-1), (row-1, col+1), (row+2, col), (row+2, col-1), (row+2, col+1), (row, col-1), (row+1, col-1), (row, col+1), (row+1, col+1)]
                        pieces_surrounding = False
                        # Verifica se alguma célula adjacente já tem peças
                        for r, c in neighbors:
                            if r < 0 or r > state.board.size_rows-1 or c < 0 or c > state.board.size_columns-1:
                                continue  # Ignorar células fora do tabuleiro
                                
                            value = state.board.get_value(r, c)
                            if value not in [None, 'W', '.']:
                                pieces_surrounding = True
                                break
                        
                        # Coloca o barco se não houver peças adjacentes
                        # Não verifica se está a escrever por cima de uma dica, as dicas são colocadas novamente após encontrar a solução   
                        if not pieces_surrounding:
                            actions_aux.extend([(row, col, 't'), (row+1, col, 'b')])        # barco

                            # Coloca as águas à volta do barco em posições válidas
                            if row > 0:
                                actions_aux.append((row-1, col, '.'))
                                if col > 0:
                                    actions_aux.append((row-1, col-1, '.'))
                                if col < 9: 
                                    actions_aux.append((row-1, col+1, '.'))

                            if row+1 < 9:
                                actions_aux.append((row+2, col, '.'))
                                if col > 0:
                                    actions_aux.append((row+2, col-1, '.'))
                                if col < 9: 
                                    actions_aux.append((row+2, col+1, '.'))

                            if col > 0:
                                actions_aux.extend([(row, col-1, '.'), (row+1, col-1, '.')])

                            if col < 9:
                                actions_aux.extend([(row, col+1, '.'), (row+1, col+1, '.')])

                        # Adicionar todas as peças a colocar à lista de ações
                        if actions_aux:
                            actions.append(actions_aux)
                            actions_aux = []

                    # 2 peças na horizontal - verifica se há espaço suficiente na linha e se as células estão vazias ou com uma dica válida
                    if col < state.board.size_columns-1 and state.board.get_value(row, col) in [None, 'L'] and state.board.get_value(row, col+1) in [None, 'R'] and state.board.boats_rows[row] - state.board.current_boats_rows[row] >= 2:                            
                        # Lista de células adjacentes
                        neighbors = [(row, col-1), (row-1, col-1), (row+1, col-1), (row, col+2), (row-1, col+2), (row+1, col+2), (row-1, col), (row-1, col+1), (row+1, col), (row+1, col+1)]
                        pieces_surrounding = False
                        # Verifica se alguma célula adjacente já tem peças
                        for r, c in neighbors:
                            if r < 0 or r > state.board.size_rows-1 or c < 0 or c > state.board.size_columns-1:
                                continue  # Ignorar células fora do tabuleiro
                                
                            value = state.board.get_value(r, c)
                            if value not in [None, 'W', '.']:
                                pieces_surrounding = True
                                break
                        
                        # Coloca o barco se não houver peças adjacentes
                        # Não verifica se está a escrever por cima de uma dica, as dicas são colocadas novamente após encontrar a solução   
                        if not pieces_surrounding:
                            actions_aux.extend([(row, col, 'l'), (row, col+1, 'r')])        # barco

                            # Coloca as águas à volta do barco em posições válidas
                            if col > 0:
                                actions_aux.append((row, col-1, '.'))
                                if row > 0:
                                    actions_aux.append((row-1, col-1, '.'))
                                if row < 9:
                                    actions_aux.append((row+1, col-1, '.'))

                            if col+1 < 9:
                                actions_aux.append((row, col+2, '.'))
                                if row > 0:
                                    actions_aux.append((row-1, col+2, '.'))
                                if row < 9:
                                    actions_aux.append((row+1, col+2, '.'))

                            if row > 0:
                                actions_aux.extend([(row-1, col, '.'), (row-1, col+1, '.')])

                            if row < 9:
                                actions_aux.extend([(row+1, col, '.'), (row+1, col+1, '.')])

                        # Adicionar todas as peças a colocar à lista de ações
                        if actions_aux:
                            actions.append(actions_aux)
                            actions_aux = []

                # Coloca os barcos de 1 peça depois de colocar o barcos de 4, 3 e 2 peças
                if state.board.size_4 == 1 and state.board.size_3 == 2 and state.board.size_2 == 3 and state.board.size_1 < 4:
                    # barco de 1 peça - verifica se há espaço suficiente na coluna e se a célula está vazia
                    if state.board.get_value(row, col) is None and state.board.boats_columns[col] >= 1:
                        # Lista de células adjacentes
                        neighbors = [(row-1, col), (row-1, col-1), (row-1, col+1), (row+1, col), (row+1, col-1), (row+1, col+1), (row, col-1), (row, col+1)]
                        pieces_surrounding = False
                        # Verifica se alguma célula adjacente já tem peças
                        for r, c in neighbors:
                            if r < 0 or r > state.board.size_rows-1 or c < 0 or c > state.board.size_columns-1:
                                continue  # Ignorar células fora do tabuleiro
                                
                            value = state.board.get_value(r, c)
                            if value not in [None, 'W', '.']:
                                pieces_surrounding = True
                                break
                        
                        # Coloca o barco se não houver peças adjacentes
                        # Não verifica se está a escrever por cima de uma dica, as dicas são colocadas novamente após encontrar a solução   
                        if not pieces_surrounding:
                            actions_aux.extend([(row, col, 'c')])       # barco

                            # Coloca as águas à volta do barco em posições válidas
                            if row > 0:
                                actions_aux.append((row-1, col, '.'))
                                if col > 0:
                                    actions_aux.append((row-1, col-1, '.'))
                                if col < 9: 
                                    actions_aux.append((row-1, col+1, '.'))

                            if row < 9:
                                actions_aux.append((row+1, col, '.'))
                                if col > 0:
                                    actions_aux.append((row+1, col-1, '.'))
                                if col < 9: 
                                    actions_aux.append((row+1, col+1, '.'))

                            if col > 0:
                                actions_aux.extend([(row, col-1, '.')])

                            if col < 9:
                                actions_aux.extend([(row, col+1, '.')])

                        # Adicionar todas as peças a colocar à lista de ações
                        if actions_aux:
                            actions.append(actions_aux)
                            actions_aux = []
        
        actions.reverse()
        return actions
    
    def replace_hints(self, state: BimaruState):
        """Coloca as dicas nas posições correspondentes e retornar o estado
        resultante."""

        for hint in state.board.hints:
            row, col, value = hint
            state.board.set_value(row, col, value)

        return state

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        # Cria um novo estado a partir do estado atual
        copy_board = copy.deepcopy(state.board)
        newstate = BimaruState(copy_board)

        pieces = 0          # Contador de peças colocadas
        boat_type = ''      # Tipo de barco (vertical ou horizontal - barcos de 1 peça contam como verticais)

        for act in action:
            row, column, value = act
            newstate.board.set_value(row, column, value)

            if value != '.':
                pieces += 1             # se não for '.' é uma peça de um barco
                
                # Guarda o tipo de barco colocado e as linhas e colunas alteradas
                if value in ['t', 'c']:         # Barcos verticais começam por 't' (barcos de 1 peça são tratados como verticais)
                    boat_type = 'vert'          
                    update_col = column
                    update_row = row

                elif value == 'l':              # Barcos horizontais começam por 'l'
                    boat_type = 'horz'
                    update_col = column
                    update_row = row
        

        # Preenche linhas e colunas que já tenham o número certo de peças com água nos espaços vazios
        newstate.board.fill_full_rows()
        newstate.board.fill_full_columns()

        # Incrementa o contador correspondente ao tipo de barco colocado
        if pieces == 4:
            newstate.board.size_4 += 1
        elif pieces == 3:
            newstate.board.size_3 += 1
        elif pieces == 2:
            newstate.board.size_2 += 1
        elif pieces == 1:
            newstate.board.size_1 += 1

        # Incrementa os contadores de peças em cada linha e coluna afetada
        if boat_type == 'vert':
            newstate.board.current_boats_columns[update_col] += pieces
            newstate.board.current_boats_rows[range(update_row, update_row+pieces)] += 1

        elif boat_type == 'horz':
            newstate.board.current_boats_columns[range(update_col, update_col+pieces)] += 1
            newstate.board.current_boats_rows[update_row] += pieces

        return newstate
        
    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # Inicializa as variáveis
        num_boat_1 = num_boat_2 = num_boat_3 = num_boat_4 = 0
        board = state.board
        
        # Se algum elemento do board estiver vazio, retorna False
        if np.any(board.board == None):
            return False

        # Verifica se o número de barcos na linha é o correto
        boats_in_row = np.sum((board.board != None) & (board.board != '.') & (board.board != 'W'), axis=1)
        if not np.all(boats_in_row == board.boats_rows):
            return False

        # Verifica se o número de barcos na linha é o correto
        boats_in_column = np.sum((board.board != None) & (board.board != '.') & (board.board != 'W'), axis=0)
        if not np.all(boats_in_column == board.boats_columns):
            return False

        # Retorna False se algumas das dicas não for usada, i.e., 
        # para dicas que não sejam nem água nem um barco de tamanho 1, 
        # verifica se alguma das posições à volta é parte de um barco. 
        # Caso nenhuma delas seja, então a dica não está a ser usada.
        for row in range(board.size_rows):
            for col in range(board.size_columns):
                cell_value = board.get_value(row, col)
                if cell_value not in [None, 'W', 'C'] and cell_value.isupper():
                    hint_validation = False
                    neighbors = np.array([(row-1, col), (row-1, col-1), (row-1, col+1), (row+1, col), (row+1, col-1), (row+1, col+1), (row, col-1), (row, col+1)])
                    for r, c in neighbors:
                        if r < 0 or r > board.size_rows-1 or c < 0 or c > board.size_columns-1:
                            continue
                                    
                        value = board.get_value(r, c)
                        if value not in [None, 'W', '.']:
                            hint_validation = True
                            break
                    else:
                        return False

        self.replace_hints(state)
        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # Inicializa as variáveis
        state = node.state
        parent = node.parent
        heuristic = 0

        # Caso seja o primeiro nó, a heuristica será zero dado que não tem pais.
        # Caso contrário a heurística será a soma do número de barcos esperados 
        # nas linhas e colunas em que o barco é colocado. Quanto maior esta soma,
        # mais provável é que o estado seja o correto.
        if parent is None:
            heuristic = 0
        else:
            index_placed_boat = np.where((parent.state.board.board != state.board.board) & (state.board.board != '.') & (state.board.board != 'W'))
            rows, cols = np.unique(index_placed_boat[0]), np.unique(index_placed_boat[1])
            heuristic -= np.sum(parent.state.board.boats_rows[rows])
            heuristic -= np.sum(parent.state.board.boats_columns[cols])
            # É adicionado um número inteiro à heurística dependendo do número de barcos colocados de forma a que 
            # a ordem das ações seja: 1 - colocar um barco de 4, 2 - um barco de 3, 3- outro barco de 3, e assim sucessivamente
            if state.board.size_4 == 1 and state.board.size_3 == 0:
                heuristic += 550
            if state.board.size_4 == 1 and state.board.size_3 > 0 and state.board.size_3 <= 2 and state.board.size_2 == 0:
                if state.board.size_3 == 1:
                    heuristic += 450
                if state.board.size_3 == 2:
                    heuristic += 350
            if state.board.size_4 == 1 and state.board.size_3 == 2 and state.board.size_2 > 0 and state.board.size_2 <= 3:
                if state.board.size_2 == 1:
                    heuristic += 250
                if state.board.size_2 == 2:
                    heuristic += 150
                if state.board.size_2 == 3:
                    heuristic += 50
            if state.board.size_4 == 1 and state.board.size_3 == 2 and state.board.size_2 == 3 and state.board.size_1 > 0 and state.board.size_1 <= 4:
                if state.board.size_1 == 1:
                    heuristic -= 50
                if state.board.size_1 == 2:
                    heuristic -= 150
                if state.board.size_1 == 3:
                    heuristic -= 250
                if state.board.size_1 == 4:
                    heuristic -= 350
        

        # Calcula o número de células vazias nas linhas, colunas e na totalidade
        empty_rows = np.sum(state.board.board == None, axis=1)
        empty_columns = np.sum(state.board.board == None, axis=0)
        num_empty_cells = np.sum(state.board.board == None)
        
        # Atribui um valor infinito à heurística (de modo a que o estado não 
        # seja considerado) se algumas das dicas for ignorada
        for row in range(state.board.size_rows):
            for col in range(state.board.size_columns):
                cell_value = state.board.get_value(row, col)
                if cell_value not in [None, 'W', 'C'] and cell_value.isupper():
                    hint_validation = False
                    neighbors = np.array([(row-1, col), (row-1, col-1), (row-1, col+1), (row+1, col), (row+1, col-1), (row+1, col+1), (row, col-1), (row, col+1)])
                    for r, c in neighbors:
                        if r < 0 or r > state.board.size_rows-1 or c < 0 or c > state.board.size_columns-1:
                            continue
                                    
                        value = state.board.get_value(r, c)
                        if value not in ['W', '.']:
                            break
                    else:    
                        heuristic = np.inf

        # Calcula o número de barcos atual nas linhas e colunas e as linhas e colunas que não têm células vazias
        boats_in_row = np.sum((state.board.board != None) & (state.board.board != '.') & (state.board.board != 'W'), axis=1)
        boats_in_column = np.sum((state.board.board != None) & (state.board.board != '.') & (state.board.board != 'W'), axis=0)
        complete_rows = (np.sum(state.board.board == None, axis=1) == 0)
        complete_columns = (np.sum(state.board.board == None, axis=0) == 0)

        # Atribui um valor infinito à heurística (de modo a que o estado não 
        # seja considerado) caso: 1 - o número de barcos a colocar nas linhas e colunas
        # seja superior ao número de espaços livres nessas linhas e colunas, 2 - Para linhas 
        # e colunas preenchidas o número de barcos esperados nessas linhas e colunas 
        # não corresponda ao número de barco existentes nas mesmas
        if np.any(state.board.boats_rows - boats_in_row > empty_rows):
            heuristic = np.inf
        elif np.any(state.board.boats_columns - boats_in_column > empty_columns):
            heuristic = np.inf
        elif np.any(state.board.boats_rows[complete_rows] != boats_in_row[complete_rows]):
            heuristic = np.inf
        elif np.any(state.board.boats_columns[complete_columns] != boats_in_column[complete_columns]):
            heuristic = np.inf

        # Calcula os índices para os quais estão colocadas dicas (excepto quando se tratam de água ou barcos de tamanho 1)
        upper_case_index = np.argwhere(np.char.isupper(state.board.board.astype(str)) & (state.board.board != 'W') & (state.board.board != 'C'))
        rows, cols = upper_case_index[:,0], upper_case_index[:,1]
        upper_letters = state.board.board[rows, cols]

        # Atribui um valor infinito à heurística (de modo a que o estado não 
        # seja considerado) caso a colocação de alguma das peças em torno 
        # das dicas não seja possível
        if np.all(upper_letters):
            for upper, row_aux, col_aux in zip(upper_letters, rows, cols):
                if upper == 'T':
                    if state.board.board[row_aux+1, col_aux] == '.':
                        heuristic = np.inf
                elif upper == 'B':
                    if state.board.board[row_aux-1, col_aux] == '.':
                        heuristic = np.inf
                elif upper == 'R':
                    if state.board.board[row_aux, col_aux-1] == '.':
                        heuristic = np.inf
                elif upper == 'L':
                    if state.board.board[row_aux, col_aux+1] == '.':
                        heuristic = np.inf
                elif upper == 'M':
                    count = 0
                    if col_aux < state.board.size_columns - 1:
                        if state.board.board[row_aux, col_aux+1] == '.':
                            count += 1
                    if col_aux > 0:
                        if state.board.board[row_aux, col_aux-1] == '.':
                            count += 1
                    if row_aux > 0:
                        if state.board.board[row_aux-1, col_aux] == '.':
                            count += 1
                    if row_aux < state.board.size_rows - 1:
                        if state.board.board[row_aux+1, col_aux] == '.':
                            count += 1
                    if count > 2:
                        heuristic = np.inf
            
        return heuristic



if __name__ == "__main__":
    # Lê o ficheiro do standard input,
    # Usa uma técnica de procura para resolver a instância,
    # Retira a solução a partir do nó resultante,
    # Imprime para o standard output no formato indicado.

    # Lê grelha dos ficheiros
    board = Board.parse_instance()
    
    # Cria uma instância de Bimaru:
    problem = Bimaru(board)

    # Obtem o nó solução usando a procura em A*:
    goal_node = astar_search(problem)

    # Imprime a solução
    goal_node.state.board.print()

