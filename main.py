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
            self.next = Token("LPAREN", self.source[self.position])
            self.position += 1
        elif self.source[self.position] == ")":
            self.next = Token("RPAREN", self.source[self.position])
            self.position += 1
        else:
            raise Exception(f"Caracter inválido: {self.source[self.position]}")
        



class Parser:
    def parse_expression() -> int:
        resultado = Parser.parse_term()
        while Parser.lex.next.kind in ("PLUS", "MINUS"):
            if Parser.lex.next.kind == "PLUS":
                Parser.lex.select_next()
                resultado += Parser.parse_term()
            elif Parser.lex.next.kind == "MINUS":
                Parser.lex.select_next()
                resultado -= Parser.parse_term()
        return resultado

    def parse_term():
        resultado = Parser.parse_factor()
        while Parser.lex.next.kind in ("MULT", "DIV"):
            if Parser.lex.next.kind == "MULT":
                Parser.lex.select_next()
                resultado *= Parser.parse_factor()
            elif Parser.lex.next.kind == "DIV":
                Parser.lex.select_next()
                resultado //= Parser.parse_factor()
        return resultado

    def parse_factor():
        if Parser.lex.next.kind == "INT":
            resultado = Parser.lex.next.value
            Parser.lex.select_next()
            return resultado
        elif Parser.lex.next.kind == "LPAREN":
            Parser.lex.select_next()
            resultado = Parser.parse_expression()
            if Parser.lex.next.kind != "RPAREN":
                raise Exception("Faltando )")
            Parser.lex.select_next()
            return resultado
        elif Parser.lex.next.kind == "MINUS":
            Parser.lex.select_next()
            return -Parser.parse_factor()
        elif Parser.lex.next.kind == "PLUS":
            Parser.lex.select_next()
            return Parser.parse_factor()
        else:
            raise Exception("Fator inválido")
        

    def run(code: str):
        Parser.lex = Lexer(code,0,None)
        Parser.lex.select_next()
        resultado = Parser.parse_expression()
        if Parser.lex.next.kind != "EOF":
            raise Exception("Expressão inválidas,,")
        return resultado
    
codigo= sys.argv[1]
print(Parser.run(codigo))