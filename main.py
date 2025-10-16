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
        
        
        elif self.source[self.position] == "\"":
            self.position += 1
            string_val = ""
            while self.position < len(self.source) and self.source[self.position] != "\"":
                string_val += self.source[self.position]
                self.position += 1
            if self.position < len(self.source) and self.source[self.position] == "\"":
                self.position += 1  # Pula a aspa de fechamento
                self.next = Token("STR", string_val)
            else:
                raise LexerError("String não fechada")

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
    def __init__(self, value: int|str|bool, type: str):
        self.value = value
        self.type = type            

class SymbolTable:
    def __init__(self,table: dict[str, Variable]):
        self.table = table

    # cria uma variável na tabela
    def create_variable(self, name: str, var: Variable):
        if name in self.table.keys():
            raise SemanticError(f"Variável '{name}' já declarada.")
        self.table[name] = var

    def set(self, var: str, value: int|str|bool, type: str):
        if var not in self.table.keys():
            raise SemanticError(f"Erro: variável '{var}' não declarada")
        if self.table[var].type != type:
            raise SemanticError(f"Erro: tipo incorreto para variável '{var}'. Esperado '{self.table[var].type}', recebido '{type}'")
        self.table[var] = Variable(value, self.table[var].type)
        
    def getter(self, var: str):
        if var not in self.table.keys():
            raise SemanticError(f"Erro: variável '{var}' não declarada")
        return Variable(self.table[var].value, self.table[var].type)
   
    def table(self):
        """Getter: retorna o dicionário de variáveis."""
        return self.table



#   get uma variável da tabela
    def get(self, name: str) -> Variable | None:
        if name not in self.table.keys():
            raise SemanticError(f"Variável '{name}' não declarada.")
        """Getter: retorna uma variável da tabela."""
        return Variable(self.table[name].value, self.table[name].type)

class Node:
    def __init__(self, value, children: list):
        self.value = value
        self.children = children

    def evaluate(self, st: SymbolTable):
        """Método abstrato que deve ser sobrescrito nas subclasses."""
        pass

class BinOp(Node):
        
    def evaluate(self, st: SymbolTable):
        l = self.children[0].evaluate(st)
        r = self.children[1].evaluate(st)
        left_val_type, left_val = l.type, l.value
        right_val_type, right_val = r.type, r.value

        def to_str(val,type):
            if type == "bool":
                return "true" if val != 0 else "false"
            return str(val)
        

        if self.value == "PLUS":
            if left_val_type == "string" or right_val_type == "string":
                return Variable(to_str(left_val,left_val_type) + to_str(right_val,right_val_type), "string")
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


class UnOp(Node):

    def evaluate(self, st: SymbolTable):
        val = self.children[0].evaluate(st)

        if self.value == "PLUS":
            if val.type != "int":
                raise SemanticError(f"Operação unária '+' inválida para '{val.type}'")
            return Variable(val.value, "int")
        elif self.value == "MINUS":
            if val.type != "int":
                raise SemanticError(f"Operação unária '-' inválida para '{val.type}'")
            return Variable(-val.value, "int")
        elif self.value == "NOT":
            if val.type != "bool":
                raise SemanticError(f"Operação unária '!' inválida para '{val.type}'")
            return Variable(0 if val.value else 1, "bool")
        else:
            raise SemanticError(f"Operador unário desconhecido: {self.value}")

class IntVal(Node):
    def evaluate(self, st: SymbolTable):
        return Variable(self.value, "int")
    

class NoOp(Node):
    def evaluate(self, st):
        pass

class Assignment(Node):
    def evaluate(self, st: SymbolTable):
        var_name = self.children[0].value
        var_value = self.children[1].evaluate(st)   # var_value é Variable
        st.set(var_name, var_value.value, var_value.type)  # << AQUI

class Identifier(Node):
    def evaluate(self, st: SymbolTable):
        return st.get(self.value)

class BoolVal(Node):
    def evaluate(self, st: SymbolTable):
        return Variable(1 if self.value else 0, "bool")

class StringVal(Node):
    def evaluate(self, st: SymbolTable):
        return Variable(self.value, "string")
    
class VarDec(Node):
    def evaluate(self, st):
        var_name = self.children[0].value
        dec_type = self.value  # "int" | "string" | "bool"
    
        if dec_type not in ("int", "string", "bool"):
            raise SemanticError(f"Tipo de variável desconhecido: {dec_type}")

        defaults = {"int": 0, "string": "", "bool": 0}

        if len(self.children) == 2:
            rhs = self.children[1].evaluate(st)  # rhs é um Variable
            if rhs.type != dec_type:
                raise SemanticError(
                    f"Tipo incompatível na inicialização da variável '{var_name}'. "
                    f"Esperado '{dec_type}', recebido '{rhs.type}'."
                )
            st.create_variable(var_name, Variable(rhs.value, dec_type))
            # opcional: já retorna a variável criada
            return Variable(rhs.value, dec_type)

        # sem inicialização
        dv = defaults[dec_type]
        st.create_variable(var_name, Variable(dv, dec_type))
        return Variable(dv, dec_type)


class Block(Node):
    def evaluate(self, st: SymbolTable):
        for stmt in self.children:
            stmt.evaluate(st)

class Print(Node):
    def evaluate(self, st: SymbolTable):
        value = self.children[0].evaluate(st)
        if value.type == "bool":
            print ("true" if value.value != 0 else "false")
        else:
            print(value.value)

class While(Node):
    def evaluate(self, st: SymbolTable):
        condicao = self.children[0].evaluate(st)
        if condicao.type != "bool":
            raise SemanticError("Condição do 'while' deve ser do tipo bool.")
        
        while self.children[0].evaluate(st).value != 0: # 0 é falso
            self.children[1].evaluate(st)

            

class If(Node):
    def evaluate(self, st: SymbolTable):
        condicao = self.children[0].evaluate(st)
        if condicao.type != "bool":
            raise SemanticError("Condição do 'if' deve ser do tipo bool.")
        
        if condicao != 0: # 0 é falso
            self.children[1].evaluate(st)
        elif len(self.children) == 3: # existe o bloco else
            self.children[2].evaluate(st)
            
class Read(Node):
    def evaluate(self, st: SymbolTable):
        value = int(input())
        return Variable(value, "int")
    




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
                if len(children) == 0:
                    raise ParserError("Erro: bloco vazio não é permitido")
                Parser.lex.select_next() # Avança para o próximo token
                return Block("BLOCK", children)
            else:
                raise ParserError("Erro: esperado '}' no final do bloco")
        else:
            raise ParserError("Erro: esperado '{' no início do bloco")
    


    def parse_statement() -> Node:
        if Parser.lex.next.kind == "END":
            folha = NoOp("NoOp", [])
            Parser.lex.select_next()
            return folha
        
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
                raise ParserError("Esperado tipo após identificador.")
            
            type_node = Parser.lex.next.value
            Parser.lex.select_next()

            expr_node = None
            if Parser.lex.next.kind == "ASSIGN":
                Parser.lex.select_next()
                expr_node = Parser.parseBoolExpression()
                
            
            if Parser.lex.next.kind != "END":
                raise ParserError("Esperado fim de linha após declaração.")
            Parser.lex.select_next()
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
                raise ParserError("Faltando )")
            Parser.lex.select_next()
            return node
        elif Parser.lex.next.kind == "BOOL":
            node = BoolVal(Parser.lex.next.value,[])
            Parser.lex.select_next()
            return node
        elif Parser.lex.next.kind == "STR":
            node = StringVal(Parser.lex.next.value,[])
            Parser.lex.select_next()
            return node
        
        elif Parser.lex.next.kind == "READ":
            Parser.lex.select_next()
            if Parser.lex.next.kind != "OPEN_PAR":
                raise ParserError("Esperado '(' após 'Scanln'.")
            Parser.lex.select_next()
            if Parser.lex.next.kind != "CLOSE_PAR":
                raise ParserError("Esperado ')' após 'Scanln('.")
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
