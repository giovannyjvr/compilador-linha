import sys
import token
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
            reservadas = {"print": "PRINT"}
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
            self.next = Token("ASSIGN", self.source[self.position])
            self.position += 1

        elif self.source[self.position] == "\n":
            self.next = Token("END", "\\n")  # ou só "\n"
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


class UnOp(Node):

    def evaluate(self, st: SymbolTable):
        val = self.children[0].evaluate(st)

        if self.value == "PLUS":
            return val
        elif self.value == "MINUS":
            return -val

class IntVal(Node):
    def evaluate(self, st: SymbolTable):
        return self.value

class NoOp():
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


class Parser:
    def parse_program() -> Block:
        statements = []
        while Parser.lex.next.kind != "EOF":
            statements.append(Parser.parse_statement())
        return Block("BLOCK",statements)

    def parse_statement() -> Node:
        if Parser.lex.next.kind == "IDEN":
            id_node = Identifier(Parser.lex.next.value, [])
            Parser.lex.select_next()
            if Parser.lex.next.kind != "ASSIGN":
                raise Exception("Esperado '=' após identificador.")
            Parser.lex.select_next()
            expr_node = Parser.parse_expression()
            if Parser.lex.next.kind != "END":
                raise Exception("Esperado fim de linha após expressão.")
            Parser.lex.select_next()
            return Assignment("ASSIGN", [id_node, expr_node])
        
        elif Parser.lex.next.kind == "PRINT":
            Parser.lex.select_next()
            expr_node = Parser.parse_expression()
            if Parser.lex.next.kind != "END":
                raise Exception("Esperado fim de linha após expressão.")
            Parser.lex.select_next()
            return Print("PRINT", [expr_node])
        elif Parser.lex.next.kind == "END":
            Parser.lex.select_next()
            return NoOp()
        else:
            raise Exception("Início de comando inválido.")

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

    def parse_factor():
        if Parser.lex.next.kind == "INT":
            node = IntVal(Parser.lex.next.value,[])
            Parser.lex.select_next()
            return node
        elif Parser.lex.next.kind == "OPEN_PAR":
            Parser.lex.select_next()
            node = Parser.parse_expression()
            if Parser.lex.next.kind != "CLOSE_PAR":
                raise Exception("Faltando )")
            Parser.lex.select_next()
            return node
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