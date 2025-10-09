import sys
import re

class Token:
    def __init__(self, kind : str, value: str| int):
        self.kind = kind #tipo do token, INT PLUS MINUS EOF
        self.value = value # valor do token "1 , 2 , 3 , 4 , + , -"

class Prepro:
    @staticmethod
    def filter(code: str) -> str:
        """
        Remove comentários inline: tudo entre '//' e fim da linha,
        preservando a quebra de linha.
        """
        # cobre \n (Unix) e \r\n (Windows) sem consumir a quebra
        return re.sub(r'//.*?(?=\r?\n|$)', '', code)
        


class Lexer:
    def __init__(self, source: str, position : int, next : Token):
        self.source = source
        self.position = position
        self.next = next

    def select_next(self):
        if self.position >= len(self.source):
            self.next = Token("EOF", "")
            return
        
        while self.position < len(self.source) and self.source[self.position] in (" ", "\t", "\r"):
            self.position += 1
            if self.position >= len(self.source):
                self.next = Token("EOF", "")
                return
            
        if self.source[self.position].isalpha():
            reservadas = {"Println": "PRINT",
                          "if": "IF",
                          "else": "ELSE",
                          "for": "WHILE",
                          "Scanln": "READ",
                          }
            
            idnt = self.source[self.position]
            self.position += 1
            while self.position < len(self.source) and (self.source[self.position].isalnum() or self.source[self.position] == "_"):
                idnt += self.source[self.position]
                self.position += 1

            identificador = idnt
            if identificador in reservadas:
                self.next = Token(reservadas[identificador], identificador)
            else:
                self.next = Token("IDEN", identificador)

        elif self.source[self.position].isdigit():
            numero = self.source[self.position]
            self.position += 1
            while self.position < len(self.source) and self.source[self.position].isdigit():
                numero += self.source[self.position]
                self.position += 1
            self.next = Token("INT", int(numero))

        elif self.source[self.position] == "-":
            self.next = Token("MINUS", self.source[self.position])
            self.position += 1

        elif self.source[self.position] == "+":
            self.next = Token("PLUS", self.source[self.position])
            self.position += 1

        elif self.source[self.position] == "*":
            self.next = Token("MULT", self.source[self.position])
            self.position += 1

        elif self.source[self.position] == "/":
            self.next = Token("DIV", self.source[self.position])
            self.position += 1

        elif self.source[self.position] == "(":
            self.next = Token("OPEN_PAR", self.source[self.position])
            self.position += 1

        elif self.source[self.position] == ")":
            self.next = Token("CLOSE_PAR", self.source[self.position])
            self.position += 1

        elif self.source[self.position] == "=":
            # Pode ser atribuição ou comparação ==
            if self.position + 1 < len(self.source) and self.source[self.position + 1] == "=":
                self.next = Token("EQUAL", "==")
                self.position += 2
            else:
                self.next = Token("ASSIGN", self.source[self.position])
                self.position += 1

        elif self.source[self.position] == "\n":
            self.next = Token("END", "\\n")  # ou só "\n"
            self.position += 1

        elif self.source[self.position] == "&":
            if self.position + 1 < len(self.source) and self.source[self.position + 1] == "&":
                self.next = Token("AND", "&&")
                self.position += 2
            else:
                raise Exception(f"Caracter inválido: {self.source[self.position]}")
            
        elif self.source[self.position] == "|":
            if self.position + 1 < len(self.source) and self.source[self.position + 1] == "|":
                self.next = Token("OR", "||")
                self.position += 2
            else:
                raise Exception(f"Caracter inválido: {self.source[self.position]}")
            
        elif self.source[self.position] == "!":
            self.next = Token("NOT", "!")
            self.position += 1
        
        # Abre e fecha chaves para blocos de código
        elif self.source[self.position] == "{":
            self.next = Token("OPEN_BRA", "{")
            self.position += 1
        elif self.source[self.position] == "}":
            self.next = Token("CLOSE_BRA", "}")
            self.position += 1

        # Operadores relacionais: <, >   || LT e GT
        elif self.source[self.position] == "<":
            self.next = Token("LT", "<")
            self.position += 1
        elif self.source[self.position] == ">":
            self.next = Token("GT", ">")
            self.position += 1
        
        
        
        else:
            raise Exception(f"Caracter inválido: {self.source[self.position]}")
        
class Variable:
    def __init__(self, value: int):
        self.value = value

class SymbolTable:
    def __init__(self):
        self._table: dict[str, Variable] = {}

    def table(self):
        """Getter: retorna o dicionário de variáveis."""
        return self._table

    def set(self, name: str, var: Variable):
        """Setter: adiciona uma variável à tabela."""
        self._table[name] = var

    def get(self, name: str) -> Variable | None:
        """Getter: retorna uma variável da tabela."""
        return self._table.get(name)


class Node:
    def __init__(self, value, children: list):
        self.value = value
        self.children = children

    def evaluate(self, st: SymbolTable):
        """Método abstrato que deve ser sobrescrito nas subclasses."""
        pass

class BinOp(Node):
        

    def evaluate(self, st: SymbolTable):
        left_val = self.children[0].evaluate(st)
        right_val = self.children[1].evaluate(st)

        if self.value == "PLUS":
            return left_val + right_val
        elif self.value == "MINUS":
            return left_val - right_val
        elif self.value == "MULT":
            return left_val * right_val
        elif self.value == "DIV":
            return left_val // right_val
        elif self.value == "EQUAL":
            return 1 if left_val == right_val else 0
        elif self.value == "LT":
            return 1 if left_val < right_val else 0
        elif self.value == "GT":   
            return 1 if left_val > right_val else 0
        elif self.value == "AND":
            return 1 if (left_val != 0 and right_val != 0) else 0
        elif self.value == "OR":
            return 1 if (left_val != 0 or right_val != 0) else 0
        else:
            raise Exception(f"Operador binário desconhecido: {self.value}")


class UnOp(Node):

    def evaluate(self, st: SymbolTable):
        val = self.children[0].evaluate(st)

        if self.value == "PLUS":
            return val
        elif self.value == "MINUS":
            return -val
        elif self.value == "NOT":
            return 1 if val == 0 else 0
        else:
            raise Exception(f"Operador unário desconhecido: {self.value}")

class IntVal(Node):
    def evaluate(self, st: SymbolTable):
        return self.value

class NoOp(Node):
    def evaluate(self, st):
        pass

class Assignment(Node):
    def evaluate(self, st: SymbolTable):
        st.set(self.children[0].value, Variable(self.children[1].evaluate(st)))

class Identifier(Node):
    def evaluate(self, st: SymbolTable):
        var = st.get(self.value)
        if var is None:
            raise Exception(f"Variável não definida: {self.value}")
        return var.value

class Block(Node):
    def evaluate(self, st: SymbolTable):
        for stmt in self.children:
            stmt.evaluate(st)

class Print(Node):
    def evaluate(self, st: SymbolTable):
        value = self.children[0].evaluate(st)
        print(value)

class While(Node):
    def evaluate(self, st: SymbolTable):
        while self.children[0].evaluate(st) != 0: # 0 é falso
            self.children[1].evaluate(st)

class If(Node):
    def evaluate(self, st: SymbolTable):
        if self.children[0].evaluate(st) != 0: # 0 é falso
            self.children[1].evaluate(st)
        elif len(self.children) == 3: # existe o bloco else
            self.children[2].evaluate(st)
            
class Read(Node):
    def evaluate(self, st: SymbolTable):
        value = int(input())
        return value   
    



class Parser:
    def parse_program() -> Block:
        statements = []
        while Parser.lex.next.kind != "EOF":
            statements.append(Parser.parse_statement())
        return Block("BLOCK",statements)
    
    def parse_block() -> Block:
        if Parser.lex.next.kind == "OPEN_BRA":
            Parser.lex.select_next() # Avança para o próximo token
            children = []
            while Parser.lex.next.kind != "CLOSE_BRA":
                node = Parser.parse_statement()
                children.append(node)
            if Parser.lex.next.kind == "CLOSE_BRA":
                Parser.lex.select_next() # Avança para o próximo token
                return Block("BLOCK", children)
            else:
                raise Exception("Erro: esperado '}' no final do bloco")
        else:
            raise Exception("Erro: esperado '{' no início do bloco")
    


    def parse_statement() -> Node:
        if Parser.lex.next.kind == "END":
            folha = NoOp("NoOp", [])
            Parser.lex.select_next()
            return folha
        
        elif Parser.lex.next.kind == "IDEN":
            id_node = Identifier(Parser.lex.next.value, [])
            Parser.lex.select_next()
            if Parser.lex.next.kind != "ASSIGN":
                raise Exception("Esperado '=' após identificador.")
            Parser.lex.select_next()
            expr_node = Parser.parseBoolExpression()
            if Parser.lex.next.kind != "END":
                raise Exception("Esperado fim de linha após expressão.")
            Parser.lex.select_next()
            return Assignment("ASSIGN", [id_node, expr_node])
        
        elif Parser.lex.next.kind == "IF":
            Parser.lex.select_next()
            cond_node = Parser.parseBoolExpression()
            if Parser.lex.next.kind == "END":
                raise Exception("Unexpected token NEWLINE")
            if Parser.lex.next.kind != "OPEN_BRA":
                raise Exception("Esperado '{' após condição do if/for")
            then_node = Parser.parse_statement()
            else_node = None
            if Parser.lex.next.kind == "ELSE":
                Parser.lex.select_next()
                if Parser.lex.next.kind == "END":
                    raise Exception("Unexpected token NEWLINE")
                if Parser.lex.next.kind != "OPEN_BRA":
                    raise Exception("Esperado '{' após else")
                else_node = Parser.parse_statement()
                return If("IF", [cond_node, then_node, else_node])
            return If("IF", [cond_node, then_node])
        
        elif Parser.lex.next.kind == "WHILE":
            Parser.lex.select_next()
            cond_node = Parser.parseBoolExpression()
            body_node = Parser.parse_block()
            return While("WHILE", [cond_node, body_node])  
        
        elif Parser.lex.next.kind == "PRINT":
            Parser.lex.select_next()
            if Parser.lex.next.kind != "OPEN_PAR":
                raise Exception("Esperado '(' após 'Println'.")
            Parser.lex.select_next()
            expr_node = Parser.parseBoolExpression()
            if Parser.lex.next.kind != "CLOSE_PAR":
                raise Exception("Esperado ')' após expressão.")
            Parser.lex.select_next()
            if Parser.lex.next.kind != "END":
                raise Exception("Esperado fim de linha após expressão.")
            Parser.lex.select_next()
            return Print("PRINT", [expr_node])
        
        elif Parser.lex.next.kind == "OPEN_BRA":
            return Parser.parse_block()
        
        else:
            raise Exception(f"Token inesperado: {Parser.lex.next.kind}")
        

    def parseBoolExpression() -> int:
        node = Parser.parseBoolTerm()
        if Parser.lex.next.kind == "OR":
            Parser.lex.select_next()
            node = BinOp("OR", [node, Parser.parseBoolTerm()])
        return node
    
    def parseBoolTerm() -> int:
        node = Parser.parser_relExpression()
        if Parser.lex.next.kind == "AND":
            Parser.lex.select_next()
            node = BinOp("AND", [node, Parser.parser_relExpression()])
        return node
    
    def parser_relExpression() -> int:
        node = Parser.parse_expression()
        if Parser.lex.next.kind in ("EQUAL", "LT", "GT"):
            op = Parser.lex.next.kind
            Parser.lex.select_next()
            node = BinOp(op, [node, Parser.parse_expression()])
        return node

    def parse_expression() -> int:
        node = Parser.parse_term()
        while Parser.lex.next.kind in ("PLUS", "MINUS"):
            if Parser.lex.next.kind == "PLUS":
                Parser.lex.select_next()
                node = BinOp("PLUS", [node, Parser.parse_term()])
            elif Parser.lex.next.kind == "MINUS":
                Parser.lex.select_next()
                node = BinOp("MINUS", [node, Parser.parse_term()])
        return node

    def parse_term():
        node = Parser.parse_factor()
        while Parser.lex.next.kind in ("MULT", "DIV"):
            if Parser.lex.next.kind == "MULT":
                Parser.lex.select_next()
                node = BinOp("MULT", [node, Parser.parse_factor()])
            elif Parser.lex.next.kind == "DIV":
                Parser.lex.select_next()
                node = BinOp("DIV", [node, Parser.parse_factor()])
        return node


    # Corrigir a função parseFactor() para incluir a operação unária NOT e a função READ.


    def parse_factor():
        if Parser.lex.next.kind == "INT":
            node = IntVal(Parser.lex.next.value,[])
            Parser.lex.select_next()
            return node
        
        elif Parser.lex.next.kind == "OPEN_PAR":
            Parser.lex.select_next()
            node = Parser.parseBoolExpression()
            if Parser.lex.next.kind != "CLOSE_PAR":
                raise Exception("Faltando )")
            Parser.lex.select_next()
            return node
        
        elif Parser.lex.next.kind == "READ":
            Parser.lex.select_next()
            if Parser.lex.next.kind != "OPEN_PAR":
                raise Exception("Esperado '(' após 'Scanln'.")
            Parser.lex.select_next()
            if Parser.lex.next.kind != "CLOSE_PAR":
                raise Exception("Esperado ')' após 'Scanln('.")
            Parser.lex.select_next()
            return Read("READ", [])
        
        elif Parser.lex.next.kind == "NOT":
            Parser.lex.select_next()
            return UnOp("NOT", [Parser.parse_factor()])
        
        elif Parser.lex.next.kind == "MINUS":
            Parser.lex.select_next()
            return UnOp("MINUS", [Parser.parse_factor()])
        
        elif Parser.lex.next.kind == "PLUS":
            Parser.lex.select_next()
            return UnOp("PLUS", [Parser.parse_factor()])
        
        elif Parser.lex.next.kind == "IDEN":
            node = Identifier(Parser.lex.next.value, [])
            Parser.lex.select_next()
            return node
        
        else:
            raise Exception("Fator inválido")
        

    @staticmethod
    def run(code: str) -> Node:
        # print(code)
        clean = Prepro.filter(code)
        # print(clean)
        Parser.lex = Lexer(clean, 0, None)
        # print(Parser.lex)
        Parser.lex.select_next()
        ast = Parser.parse_program()
        if Parser.lex.next.kind != "EOF":
            raise Exception("Tokens remanescentes após o fim do programa.")
        return ast
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 main.py teste.go")
        sys.exit(1)
    filename = sys.argv[1]
    with open(filename, "r", encoding="utf-8") as f:
        code = f.read()
    root = Parser.run(code)
    st = SymbolTable()
    root.evaluate(st)