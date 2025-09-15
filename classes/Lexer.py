from token import Token

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
            
        if self.source[self.position].isalpha():
            idnt = self.source[self.position]
            while self.position < len(self.source) and (self.source[self.position].isalnum() or self.source[self.position] == "_"):
                idnt += self.source[self.position]
                self.position += 1

            identificador = idnt
            if identificador == "print":
                self.next = Token("PRINT", identificador)
            else:
                self.next = Token("ID", identificador)

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

        elif self.source[self.position] == "=":
            self.next = Token("ASSIGN", self.source[self.position])
            self.position += 1
        
        else:
            raise Exception(f"Caracter inválido: {self.source[self.position]}")