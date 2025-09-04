import sys
import token

class Token:
    def __init__(self, kind : str, value: str| int):
        self.kind = kind #tipo do token, INT PLUS MINUS EOF
        self.value = value # valor do token "1 , 2 , 3 , 4 , + , -"

class Lexer:
    def __init__(self, source: str, position : int, next : Token):
        self.source = source
        self.position = position
        self.next = next

    def select_next(self):
        if self.position >= len(self.source):
            self.next = Token("EOF", "")
            return
        
        while self.source[self.position] == " " :
            self.position += 1  
            if self.position >= len(self.source):
                self.next = Token("EOF", "")
                return

        if self.source[self.position].isdigit():
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
        else:
            raise Exception(f"Caracter inválido: {self.source[self.position]}")
        

class Node:
    def __init__(self, value, children: list):
        self.value = value
        self.children = children

    def evaluate(self):
        """Método abstrato que deve ser sobrescrito nas subclasses."""
        pass

class BinOp(Node):
          

    def evaluate(self):
        left_val = self.children[0].evaluate()
        right_val = self.children[1].evaluate()

        if self.value == "PLUS":
            return left_val + right_val
        elif self.value == "MINUS":
            return left_val - right_val
        elif self.value == "MULT":
            return left_val * right_val
        elif self.value == "DIV":
            return left_val // right_val


class UnOp(Node):
    
    def evaluate(self):
        val = self.children[0].evaluate()

        if self.value == "PLUS":
            return val
        elif self.value == "MINUS":
            return -val

class IntVal(Node):
    def evaluate(self):
        return self.value

class Parser:
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
        else:
            raise Exception("Fator inválido")
        

    def run(code: str):
        Parser.lex = Lexer(code,0,None)
        Parser.lex.select_next()
        resultado = Parser.parse_expression()
        if Parser.lex.next.kind != "EOF":
            raise Exception("Expressão inválidas,,")
        return resultado.evaluate()
    
codigo= sys.argv[1]
print(Parser.run(codigo))