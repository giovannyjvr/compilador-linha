import sys
import re
class LexerError(Exception): pass
class ParserError(Exception): pass
class SemanticError(Exception): pass

class Token:
    def __init__(self, kind : str, value: str| int):
        self.kind = kind #tipo do token, INT PLUS MINUS EOF
        self.value = value # valor do token "1 , 2 , 3 , 4 , + , -"

class Prepro:
    @staticmethod
    def filter(code: str) -> str:
        code = re.sub(r'//.*?(?=\r?\n|$)', '', code)
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        if not code.endswith('\n'):
            code += '\n'
        return code

class Code:

    instructions = []

    def append(code: str) -> None:
        Code.instructions.append(code)

    def dump(filename: str) -> None:
        with open(filename, 'w') as file:
            # Escrever o cabeçalho: até início do código gerado            
            file.write("section .data\n")
            file.write("format_out: db \"%d\", 10, 0 ; format do printf\n")
            file.write("format_in: db \"%d\", 0 ; format do scanf\n")
            file.write("scan_int: dd 0; 32-bits integer\n\n")
            file.write("section .text\n\n")
            file.write("extern printf ; usar _printf para Windows\n")
            file.write("extern scanf ; usar _scanf para Windows\n")
            file.write("global _start ; início do programa\n")
            file.write("_start:\n")
            file.write("push ebp ; guarda o EBP\n")
            file.write("mov ebp, esp ; zera a pilha\n")

            # Escreve as instruções armazenadas
            file.write("\n".join(Code.instructions))

            # Escrever as instruções finais: após término do código gerado            
            file.write("\nmov esp, ebp ; reestabelece a pilha\n")
            file.write("pop ebp\n\n")
            file.write("; chamada da interrupcao de saida (Linux)\n")
            file.write("mov eax, 1   \n")
            file.write("xor ebx, ebx \n")
            file.write("int 0x80     \n")
            file.write("; Para Windows:\n")
            file.write("; push dword 0        \n")
            file.write("; call _ExitProcess@4\n")

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
            elif identificador == "true":
                self.next = Token("BOOL", 1)
            elif identificador == "false":
                self.next = Token("BOOL", 0)
            elif identificador in ("int", "string", "bool"):
                self.next = Token("TYPE", identificador)
            elif identificador == "var":
                self.next = Token("VAR", identificador)
            else:
                self.next = Token("IDEN", identificador)
        
        

        elif self.source[self.position].isdigit():
            number = ""
            while self.position < len(self.source) and self.source[self.position].isdigit():
                number += self.source[self.position]
                self.position += 1
            self.next = Token("INT", int(number))
        
        elif self.source[self.position] == "+":
            self.next = Token("PLUS", self.source[self.position])
            self.position += 1

        elif self.source[self.position] == "-":
            self.next = Token("MINUS", self.source[self.position])
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

        elif self.source[self.position] == "\"":
            self.position += 1  # Pula o primeiro "
            string_value = ""
            while self.position < len(self.source) and self.source[self.position] != "\"":
                string_value += self.source[self.position]
                self.position += 1
            if self.position >= len(self.source):
                raise LexerError("String malformada: faltando aspas de fechamento.")
            self.position += 1  # Pula o último "
            self.next = Token("STRING", string_value)

        elif self.source[self.position] == "\n":
            self.next = Token("END", "\\n")  # ou só "\n"
            self.position += 1

        elif self.source[self.position] == "&":
            if self.position + 1 < len(self.source) and self.source[self.position + 1] == "&":
                self.next = Token("AND", "&&")
                self.position += 2
            else:
                raise LexerError(f"Caracter inválido: {self.source[self.position]}")
            
        elif self.source[self.position] == "|":
            if self.position + 1 < len(self.source) and self.source[self.position + 1] == "|":
                self.next = Token("OR", "||")
                self.position += 2
            else:
                raise LexerError(f"Caracter inválido: {self.source[self.position]}")
            
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
            raise LexerError(f"Caracter inválido: {self.source[self.position]}")
        


class Variable:
    def __init__(self, value, type: str, shift: int = 0):
        self.value = value
        self.type = type
        self.shift = shift  # deslocamento na pilha (em bytes)


class SymbolTable:
    def __init__(self, table: dict[str, Variable]):
        self.table = table
        self.next_shift = 0  # controla o próximo deslocamento livre (em bytes)

    def create_variable(self, name: str, type: str):
        """
        Cria uma variável nova na tabela, com deslocamento na pilha.
        Lógica idêntica ao Roteiro 8: cada variável ocupa 4 bytes e
        o shift cresce de 4 em 4.
        """
        if name in self.table.keys():
            raise SemanticError(f"Variável '{name}' já declarada.")
        self.next_shift += 4
        self.table[name] = Variable(None, type, self.next_shift)

    def set(self, var: str, value, type: str):
        if var not in self.table.keys():
            raise SemanticError(f"Erro: variável '{var}' não declarada")
        if self.table[var].type != type:
            raise SemanticError(
                f"Erro: tipo incorreto para variável '{var}'. "
                f"Esperado '{self.table[var].type}', recebido '{type}'"
            )
        old = self.table[var]
        # mantém o shift original
        self.table[var] = Variable(value, old.type, old.shift)

    def getter(self, var: str) -> Variable:
        if var not in self.table.keys():
            raise SemanticError(f"Erro: variável '{var}' não declarada")
        v = self.table[var]
        return Variable(v.value, v.type, v.shift)

    # compat: se algum código usa get()
    def get(self, name: str) -> Variable:
        return self.getter(name)


class Node:
    id = 0

    @staticmethod
    def new_id() -> int:
        Node.id += 1
        return Node.id

    def __init__(self, value, children: list):
        self.value = value
        self.children = children
        self.id = Node.new_id()

    def evaluate(self, st: SymbolTable):
        pass

    def generate(self, st: SymbolTable):
        pass


class BinOp(Node):
    def evaluate(self, st: SymbolTable):
        l = self.children[0].evaluate(st)  # Variable
        r = self.children[1].evaluate(st)  # Variable
        left_val_type, left_val = l.type, l.value
        right_val_type, right_val = r.type, r.value

        def to_str(val, t):
            if t == "bool":
                return "true" if val != 0 else "false"
            return str(val)

        if self.value == "PLUS":
            if left_val_type == "string" or right_val_type == "string":
                return Variable(to_str(left_val, left_val_type) + to_str(right_val, right_val_type), "string")
            if left_val_type == right_val_type == "int":
                return Variable(left_val + right_val, "int")
            raise SemanticError(f"Operação '+' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "MINUS":
            if left_val_type == right_val_type == "int":
                return Variable(left_val - right_val, "int")
            raise SemanticError(f"Operação '-' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "MULT":
            if left_val_type == right_val_type == "int":
                return Variable(left_val * right_val, "int")
            raise SemanticError(f"Operação '*' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "DIV":
            if left_val_type == right_val_type == "int":
                if right_val == 0:
                    raise SemanticError("Divisão por zero.")
                return Variable(left_val // right_val, "int")
            raise SemanticError(f"Operação '/' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "EQUAL":
            if left_val_type != right_val_type:
                raise SemanticError(f"Operação '==' inválida entre '{left_val_type}' e '{right_val_type}'")
            return Variable(1 if left_val == right_val else 0, "bool")

        elif self.value == "LT":
            if left_val_type == right_val_type == "int":
                return Variable(1 if left_val < right_val else 0, "bool")
            if left_val_type == right_val_type == "string":
                return Variable(1 if str(left_val) < str(right_val) else 0, "bool")
            raise SemanticError(f"Operação '<' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "GT":
            if left_val_type == right_val_type == "int":
                return Variable(1 if left_val > right_val else 0, "bool")
            if left_val_type == right_val_type == "string":
                return Variable(1 if str(left_val) > str(right_val) else 0, "bool")
            raise SemanticError(f"Operação '>' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "AND":
            if left_val_type == right_val_type == "bool":
                return Variable(1 if (left_val != 0 and right_val != 0) else 0, "bool")
            raise SemanticError(f"Operação '&&' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "OR":
            if left_val_type == right_val_type == "bool":
                return Variable(1 if (left_val != 0 or right_val != 0) else 0, "bool")
            raise SemanticError(f"Operação '||' inválida entre '{left_val_type}' e '{right_val_type}'")
        else:
            raise SemanticError(f"Operador binário desconhecido: {self.value}")

    def generate(self, st: SymbolTable):
        op = self.value

        # direita
        self.children[1].generate(st)
        Code.append("push eax")
        # esquerda
        self.children[0].generate(st)
        Code.append("pop ecx")

        if op == "PLUS":
            Code.append("add eax, ecx")
        elif op == "MINUS":
            Code.append("sub eax, ecx")
        elif op == "MULT":
            Code.append("imul ecx")
        elif op == "DIV":
            Code.append("cdq")
            Code.append("idiv ecx")
        elif op == "AND":
            Code.append("cmp eax, 0")
            Code.append("mov eax, 0")
            Code.append("mov edx, 1")
            Code.append("cmovne eax, edx")
            Code.append("cmp ecx, 0")
            Code.append("mov ecx, 0")
            Code.append("mov edx, 1")
            Code.append("cmovne ecx, edx")
            Code.append("and eax, ecx")
        elif op == "OR":
            Code.append("cmp eax, 0")
            Code.append("mov eax, 0")
            Code.append("mov edx, 1")
            Code.append("cmovne eax, edx")
            Code.append("cmp ecx, 0")
            Code.append("mov ecx, 0")
            Code.append("mov edx, 1")
            Code.append("cmovne ecx, edx")
            Code.append("or eax, ecx")
        elif op in ("EQUAL", "GT", "LT"):
            Code.append("cmp eax, ecx")
            Code.append("mov eax, 0")
            Code.append("mov ecx, 1")
            if op == "EQUAL":
                Code.append("cmove eax, ecx")
            elif op == "GT":
                Code.append("cmovg eax, ecx")
            elif op == "LT":
                Code.append("cmovl eax, ecx")
        else:
            raise SemanticError(f"Operador inválido em generate(): {op}")


class UnOp(Node):
    def evaluate(self, st: SymbolTable):
        child_val = self.children[0].evaluate(st)

        if self.value == "PLUS":
            if child_val.type != "int":
                raise SemanticError("Operador unário '+' exige int")
            return Variable(+child_val.value, "int")
        elif self.value == "MINUS":
            if child_val.type != "int":
                raise SemanticError("Operador unário '-' exige int")
            return Variable(-child_val.value, "int")
        elif self.value == "NOT":
            if child_val.type != "bool":
                raise SemanticError("Operador '!' exige bool")
            return Variable(0 if child_val.value else 1, "bool")
        else:
            raise SemanticError(f"Operador unário inválido '{self.value}'")

    def generate(self, st: SymbolTable):
        self.children[0].generate(st)
        if self.value == "PLUS":
            Code.append("; unary plus (noop)")
        elif self.value == "MINUS":
            Code.append("neg eax ; unary minus")
        elif self.value == "NOT":
            Code.append("cmp eax, 0")
            Code.append("mov ecx, 1")
            Code.append("mov eax, 0")
            Code.append("cmove eax, ecx ; !bool")
        else:
            raise SemanticError(f"Operador unário inválido em generate(): {self.value}")


class IntVal(Node):
    def evaluate(self, st: SymbolTable):
        return Variable(self.value, "int")

    def generate(self, st: SymbolTable):
        Code.append(f"mov eax, {self.value} ; IntVal")


class BoolVal(Node):
    def evaluate(self, st: SymbolTable):
        return Variable(1 if self.value else 0, "bool")

    def generate(self, st: SymbolTable):
        v = 1 if self.value else 0
        Code.append(f"mov eax, {v} ; BoolVal")


class StringVal(Node):
    def evaluate(self, st: SymbolTable):
        return Variable(self.value, "string")

    def generate(self, st: SymbolTable):
        raise SemanticError("Geração de código para 'string' não suportada no Roteiro 8.")


class Identifier(Node):
    def evaluate(self, st: SymbolTable):
        return st.getter(self.value)

    def generate(self, st: SymbolTable):
        var = st.getter(self.value)
        if var.shift is None:
            raise SemanticError("Identifier sem shift na pilha.")
        Code.append(f"mov eax, [ebp-{var.shift}] ; {self.value}")


class Print(Node):
    def evaluate(self, st: SymbolTable):
        value = self.children[0].evaluate(st)
        if value.type == "bool":
            print("true" if value.value != 0 else "false")
        else:
            print(value.value)

    def generate(self, st: SymbolTable):
        self.children[0].generate(st)  # resultado em EAX
        Code.append("push eax")
        Code.append("push format_out")
        Code.append("call printf")
        Code.append("add esp, 8")


class Read(Node):
    def evaluate(self, st: SymbolTable):
        value = int(input())
        return Variable(value, "int")

    def generate(self, st: SymbolTable):
        Code.append("push scan_int")
        Code.append("push format_in")
        Code.append("call scanf")
        Code.append("add esp, 8")
        Code.append("mov eax, dword [scan_int] ; Read()")


class While(Node):
    def evaluate(self, st: SymbolTable):
        condicao = self.children[0].evaluate(st)
        if condicao.type != "bool":
            raise SemanticError("Condição do 'while' deve ser do tipo bool.")
        while self.children[0].evaluate(st).value != 0:
            self.children[1].evaluate(st)

    def generate(self, st: SymbolTable):
        my = self.id
        Code.append(f"loop_{my}:")
        self.children[0].generate(st)  # condição -> EAX
        Code.append("cmp eax, 0")
        Code.append(f"je exit_{my}")
        self.children[1].generate(st)  # corpo
        Code.append(f"jmp loop_{my}")
        Code.append(f"exit_{my}:")


class If(Node):
    def evaluate(self, st: SymbolTable):
        condicao = self.children[0].evaluate(st)
        if condicao.type != "bool":
            raise SemanticError("Condição do 'if' deve ser do tipo bool.")
        if condicao.value != 0:
            self.children[1].evaluate(st)
        elif len(self.children) == 3:
            self.children[2].evaluate(st)

    def generate(self, st: SymbolTable):
        my = self.id
        self.children[0].generate(st)  # cond em EAX
        Code.append("cmp eax, 0")

        if len(self.children) == 2:
            Code.append(f"je exit_{my}")
            self.children[1].generate(st)  # then
            Code.append(f"exit_{my}:")
        else:
            Code.append(f"je else_{my}")
            self.children[1].generate(st)  # then
            Code.append(f"jmp exit_{my}")
            Code.append(f"else_{my}:")
            self.children[2].generate(st)  # else
            Code.append(f"exit_{my}:")


class VarDec(Node):
    def evaluate(self, st: SymbolTable):
        var_name = self.children[0].value
        dec_type = self.value  # "int" | "string" | "bool"

        if dec_type not in ("int", "string", "bool"):
            raise SemanticError(f"Tipo de variável desconhecido: {dec_type}")

        defaults = {"int": 0, "string": "", "bool": 0}

        if len(self.children) == 2:
            rhs = self.children[1].evaluate(st)
            if rhs.type != dec_type:
                raise SemanticError(
                    f"Tipo incompatível na inicialização da variável '{var_name}'. "
                    f"Esperado '{dec_type}', recebido '{rhs.type}'."
                )
            st.create_variable(var_name, dec_type)
            st.set(var_name, rhs.value, rhs.type)
            return Variable(rhs.value, dec_type)

        dv = defaults[dec_type]
        st.create_variable(var_name, dec_type)
        st.set(var_name, dv, dec_type)
        return Variable(dv, dec_type)

    def generate(self, st: SymbolTable):
        var_name = self.children[0].value
        dec_type = self.value  # "int" | "bool" | "string"

        if dec_type == "string":
            raise SemanticError("Strings não têm geração de código neste roteiro.")

        # reserva 4 bytes na pilha
        Code.append("sub esp, 4 ; aloca espaço para variável")

        # registra variável na tabela
        st.create_variable(var_name, dec_type)
        var = st.getter(var_name)

        if len(self.children) == 2:
            self.children[1].generate(st)
            Code.append(f"mov [ebp-{var.shift}], eax ; {var_name} = expr (init)")
        else:
            Code.append(f"mov dword [ebp-{var.shift}], 0 ; {var_name} = 0 (default)")


class Assignment(Node):
    def evaluate(self, st: SymbolTable):
        var_name = self.children[0].value
        var_value = self.children[1].evaluate(st)
        st.set(var_name, var_value.value, var_value.type)

    def generate(self, st: SymbolTable):
        var_name = self.children[0].value
        var = st.getter(var_name)
        self.children[1].generate(st)
        Code.append(f"mov [ebp-{var.shift}], eax ; {var_name} = expr")


class Block(Node):
    def evaluate(self, st: SymbolTable):
        for stmt in self.children:
            stmt.evaluate(st)

    def generate(self, st: SymbolTable):
        for stmt in self.children:
            stmt.generate(st)


class NoOp(Node):
    def evaluate(self, st: SymbolTable):
        pass

    def generate(self, st: SymbolTable):
        pass


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
                Parser.lex.select_next() # Consome o "} "
                if Parser.lex.next.kind == "END":
                    Parser.lex.select_next() # Consome o "\n"
                else:
                    raise ParserError("Erro: esperado fim de linha após bloco.")
                return Block("BLOCK", children)
            else:
                raise ParserError("Erro: '}' esperado ao final do bloco.")
        else:
            raise ParserError("Erro: '{' esperado para iniciar um bloco.")
        
    def parse_statement() -> Node:
        if Parser.lex.next.kind == "END":
            Parser.lex.select_next()
            return NoOp("NOOP",[])
        
        elif Parser.lex.next.kind == "IDEN":
            id_node = Identifier(Parser.lex.next.value, [])
            Parser.lex.select_next()
            if Parser.lex.next.kind != "ASSIGN":
                raise ParserError("Esperado '=' após identificador.")
            Parser.lex.select_next()
            expr_node = Parser.parseBoolExpression()
            if Parser.lex.next.kind != "END":
                raise ParserError("Esperado fim de linha após expressão.")
            Parser.lex.select_next()
            return Assignment("ASSIGN", [id_node, expr_node])
        
        elif Parser.lex.next.kind == "IF":
            Parser.lex.select_next()
            cond_node = Parser.parseBoolExpression()
            if Parser.lex.next.kind == "END":
                raise ParserError("Unexpected token NEWLINE")
            then_node = Parser.parse_statement()
            else_node = None
            if Parser.lex.next.kind == "ELSE":
                Parser.lex.select_next()
                if Parser.lex.next.kind == "END":
                    raise ParserError("Unexpected token NEWLINE")
                else_node = Parser.parse_statement()
                return If("IF", [cond_node, then_node, else_node])
            return If("IF", [cond_node, then_node])

        elif Parser.lex.next.kind == "VAR":
            Parser.lex.select_next()
            if Parser.lex.next.kind != "IDEN":
                raise ParserError("Esperado identificador após 'var'.")
            id_node = Identifier(Parser.lex.next.value, [])
            Parser.lex.select_next()
            if Parser.lex.next.kind != "TYPE":
                raise ParserError("Esperado tipo da variável após identificador.")
            type_node = Parser.lex.next.value  # "int", "string", "bool"
            Parser.lex.select_next()
            expr_node = None
            if Parser.lex.next.kind == "ASSIGN":
                Parser.lex.select_next()
                expr_node = Parser.parseBoolExpression()
            if Parser.lex.next.kind != "END":
                raise ParserError("Esperado fim de linha após declaração de variável.")
            Parser.lex.select_next()
            # Cria o nó VarDec. Se existir inicialização, ela vai como child.
            return VarDec(type_node, [id_node] + ([expr_node] if expr_node else []))
        
        elif Parser.lex.next.kind == "WHILE":
            Parser.lex.select_next()
            cond_node = Parser.parseBoolExpression()
            if Parser.lex.next.kind == "END":
                raise ParserError("Unexpected token NEWLINE")
            body_node = Parser.parse_block()
            return While("WHILE", [cond_node, body_node])  
        
        elif Parser.lex.next.kind == "PRINT":
            Parser.lex.select_next()
            if Parser.lex.next.kind != "OPEN_PAR":
                raise ParserError("Esperado '(' após 'Println'.")
        
            Parser.lex.select_next()
            expr_node = Parser.parseBoolExpression()
            if Parser.lex.next.kind != "CLOSE_PAR":
                raise ParserError("Esperado ')' após expressão.")
            Parser.lex.select_next()
            if Parser.lex.next.kind != "END":
                raise ParserError("Esperado fim de linha após expressão.")
            Parser.lex.select_next()
            return Print("PRINT", [expr_node])
        
        elif Parser.lex.next.kind == "OPEN_BRA":
            return Parser.parse_block()
        
        else:
            raise ParserError(f"Token inesperado: {Parser.lex.next.kind}")
        
    def parseBoolExpression():
        node = Parser.parse_expression()
        while Parser.lex.next.kind in ("EQUAL", "LT", "GT"):
            if Parser.lex.next.kind == "EQUAL":
                Parser.lex.select_next()
                node = BinOp("EQUAL", [node, Parser.parse_expression()])
            elif Parser.lex.next.kind == "LT":
                Parser.lex.select_next()
                node = BinOp("LT", [node, Parser.parse_expression()])
            elif Parser.lex.next.kind == "GT":
                Parser.lex.select_next()
                node = BinOp("GT", [node, Parser.parse_expression()])
        return node
        
    
    def parse_expression():
        node = Parser.parse_term()
        while Parser.lex.next.kind in ("PLUS","MINUS","AND","OR"):
            if Parser.lex.next.kind == "PLUS":
                Parser.lex.select_next()
                node = BinOp("PLUS", [node, Parser.parse_term()])
            elif Parser.lex.next.kind == "MINUS":
                Parser.lex.select_next()
                node = BinOp("MINUS", [node, Parser.parse_term()])
            elif Parser.lex.next.kind == "AND":
                Parser.lex.select_next()
                node = BinOp("AND", [node, Parser.parse_term()])
            elif Parser.lex.next.kind == "OR":
                Parser.lex.select_next()
                node = BinOp("OR", [node, Parser.parse_term()])
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
                raise ParserError("Faltando )")
            Parser.lex.select_next()
            return node
        
        elif Parser.lex.next.kind == "IDEN":
            node = Identifier(Parser.lex.next.value, [])
            Parser.lex.select_next()
            return node
        
        elif Parser.lex.next.kind == "STRING":
            node = StringVal(Parser.lex.next.value,[])
            Parser.lex.select_next()
            return node
        
        elif Parser.lex.next.kind == "BOOL":
            node = BoolVal(Parser.lex.next.value, [])
            Parser.lex.select_next()
            return node
        
        elif Parser.lex.next.kind == "READ":
            Parser.lex.select_next()
            if Parser.lex.next.kind != "OPEN_PAR":
                raise ParserError("Faltando '(' após Read.")
            Parser.lex.select_next()
            if Parser.lex.next.kind != "CLOSE_PAR":
                raise ParserError("Faltando ')' após Read.")
            Parser.lex.select_next()
            return Read("READ", [])
        
        elif Parser.lex.next.kind == "PLUS":
            Parser.lex.select_next()
            return UnOp("PLUS", [Parser.parse_factor()])
        
        elif Parser.lex.next.kind == "MINUS":
            Parser.lex.select_next()
            return UnOp("MINUS", [Parser.parse_factor()])
        
        elif Parser.lex.next.kind == "NOT":
            Parser.lex.select_next()
            return UnOp("NOT", [Parser.parse_factor()])
        
        else:
            raise ParserError(f"Erro: token inesperado '{Parser.lex.next.kind}'")
        

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
            raise ParserError("Tokens remanescentes após o fim do programa.")
        
        return ast
 

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 main.py teste.go")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()

        # mantém seu filtro (pode incluir o '\n' final se quiser)
        code = Prepro.filter(code)

        # --- fase de PARSE ---
        try:
            root = Parser.run(code)
        except LexerError as e:
            print(f"[Lexer] {e}")
            sys.exit(1)
        except ParserError as e:
            print(f"[Parser] {e}")
            sys.exit(1)
        except Exception as e:
            # qualquer outra exceção durante parsing: trate como Parser
            print(f"[Parser] {e}")
            sys.exit(1)

        # --- fase SEMÂNTICA/EXECUÇÃO ---
        try:
            st = SymbolTable({})
            root.evaluate(st)
        except SemanticError as e:
            print(f"[Semantic] {e}")
            sys.exit(1)
        except Exception as e:
            # exceções em evaluate() são semânticas por convenção aqui
            print(f"[Semantic] {e}")
            sys.exit(1)

    except Exception as e:
        # fallback de segurança (I/O etc.)
        print(f"[Parser] {e}")
        sys.exit(1)
