import sys
import re

# =============================
# Tokens e Pré-processamento
# =============================

class Token:
    def __init__(self, kind: str, value: str | int):
        self.kind = kind
        self.value = value

class Prepro:
    @staticmethod
    def filter(code: str) -> str:
        """
        Remove comentários inline: tudo entre '//' e fim da linha,
        preservando a quebra de linha.
        """
        return re.sub(r'//.*?(?=\r?\n|$)', '', code)


class Lexer:
    def __init__(self, source: str, position: int, next: Token | None):
        self.source = source
        self.position = position
        self.next = next

    def select_next(self):
        if self.position >= len(self.source):
            self.next = Token("EOF", "")
            return

        # pular espaços em branco, tabs, \r
        while self.position < len(self.source) and self.source[self.position] in (" ", "\t", "\r"):
            self.position += 1
            if self.position >= len(self.source):
                self.next = Token("EOF", "")
                return

        if self.position >= len(self.source):
            self.next = Token("EOF", "")
            return

        ch = self.source[self.position]

        # palavras reservadas / identificadores
        if ch.isalpha() or ch == "_":
            reservadas = {
                "Println": "PRINT",
                "if": "IF",
                "else": "ELSE",
                "for": "WHILE",
                "Scanln": "READ",
                "func": "FUNC",        # Roteiro 9
                "return": "RETURN",    # Roteiro 9
            }

            idnt = self.source[self.position]
            self.position += 1
            while self.position < len(self.source) and (self.source[self.position].isalnum() or self.source[self.position] == "_"):
                idnt += self.source[self.position]
                self.position += 1

            if idnt in reservadas:
                self.next = Token(reservadas[idnt], idnt)
            elif idnt == "true":
                self.next = Token("BOOL", 1)
            elif idnt == "false":
                self.next = Token("BOOL", 0)
            elif idnt in ("int", "string", "bool"):
                self.next = Token("TYPE", idnt)
            elif idnt == "var":
                self.next = Token("VAR", idnt)
            else:
                self.next = Token("IDEN", idnt)
            return

        # strings
        if ch == "\"":
            self.position += 1
            string_val = ""
            while self.position < len(self.source) and self.source[self.position] != "\"":
                string_val += self.source[self.position]
                self.position += 1
            if self.position < len(self.source) and self.source[self.position] == "\"":
                self.position += 1
                self.next = Token("STR", string_val)
                return
            raise Exception("String não fechada")

        # números
        if ch.isdigit():
            numero = ch
            self.position += 1
            while self.position < len(self.source) and self.source[self.position].isdigit():
                numero += self.source[self.position]
                self.position += 1
            self.next = Token("INT", int(numero))
            return

        # operadores simples
        if ch == "-":
            self.next = Token("MINUS", ch)
            self.position += 1
            return

        if ch == "+":
            self.next = Token("PLUS", ch)
            self.position += 1
            return

        if ch == "*":
            self.next = Token("MULT", ch)
            self.position += 1
            return

        if ch == "/":
            self.next = Token("DIV", ch)
            self.position += 1
            return

        if ch == "(":
            self.next = Token("OPEN_PAR", ch)
            self.position += 1
            return

        if ch == ")":
            self.next = Token("CLOSE_PAR", ch)
            self.position += 1
            return

        if ch == "{":
            self.next = Token("OPEN_BRA", ch)
            self.position += 1
            return

        if ch == "}":
            self.next = Token("CLOSE_BRA", ch)
            self.position += 1
            return

        if ch == ",":
            self.next = Token("COMMA", ch)  # Roteiro 9
            self.position += 1
            return

        # comparação/atribuição
        if ch == "=":
            if self.position + 1 < len(self.source) and self.source[self.position + 1] == "=":
                self.next = Token("EQUAL", "==")
                self.position += 2
            else:
                self.next = Token("ASSIGN", ch)
                self.position += 1
            return

        # relacionais
        if ch == "<":
            self.next = Token("LT", "<")
            self.position += 1
            return

        if ch == ">":
            self.next = Token("GT", ">")
            self.position += 1
            return

        # lógicos
        if ch == "&":
            if self.position + 1 < len(self.source) and self.source[self.position + 1] == "&":
                self.next = Token("AND", "&&")
                self.position += 2
                return
            raise Exception(f"Caracter inválido: {ch}")

        if ch == "|":
            if self.position + 1 < len(self.source) and self.source[self.position + 1] == "|":
                self.next = Token("OR", "||")
                self.position += 2
                return
            raise Exception(f"Caracter inválido: {ch}")

        if ch == "!":
            self.next = Token("NOT", "!")
            self.position += 1
            return

        # fim de linha
        if ch == "\n":
            self.next = Token("END", "\\n")
            self.position += 1
            return

        raise Exception(f"Caracter inválido: {ch}")


# =============================
# Símbolos e Variáveis
# =============================

class Variable:
    def __init__(self, value: int | str | bool | object, type: str, is_func: bool = False):
        self.value = value          # para função, guarda o nó FuncDec
        self.type = type            # "int"|"string"|"bool"|"void"
        self.is_func = is_func      # True se isto representa uma função


class SymbolTable:
    def __init__(self, table: dict[str, Variable], parent: 'SymbolTable' = None):
        self.table = table
        self.parent = parent

    def create_variable(self, name: str, var: Variable):
        if name in self.table:
            raise Exception(f"Variável/função '{name}' já declarada neste escopo.")
        self.table[name] = var

    def _resolve(self, name: str) -> 'SymbolTable | None':
        if name in self.table:
            return self
        if self.parent is not None:
            return self.parent._resolve(name)
        return None

    def set(self, var: str, value: int | str | bool, type: str):
        scope = self._resolve(var)
        if scope is None:
            raise Exception(f"Erro: variável '{var}' não declarada")
        if scope.table[var].is_func:
            raise Exception(f"Erro: '{var}' é função, não variável")
        if scope.table[var].type != type:
            raise Exception(f"Erro: tipo incorreto para variável '{var}'. Esperado '{scope.table[var].type}', recebido '{type}'")
        scope.table[var] = Variable(value, scope.table[var].type)

    def getter(self, var: str):
        scope = self._resolve(var)
        if scope is None:
            raise Exception(f"Erro: variável/função '{var}' não declarada")
        v = scope.table[var]
        return Variable(v.value, v.type, v.is_func)

    def get(self, name: str) -> Variable:
        return self.getter(name)


# =============================
# AST Nodes + Runtime
# =============================

class Node:
    def __init__(self, value, children: list):
        self.value = value
        self.children = children

    def evaluate(self, st: SymbolTable):
        pass


class BinOp(Node):
    def evaluate(self, st: SymbolTable):
        l = self.children[0].evaluate(st)
        r = self.children[1].evaluate(st)
        left_val_type, left_val = l.type, l.value
        right_val_type, right_val = r.type, r.value

        def to_str(val, typ):
            if typ == "bool":
                return "true" if val != 0 else "false"
            return str(val)

        if self.value == "PLUS":
            if left_val_type == "string" or right_val_type == "string":
                return Variable(to_str(left_val, left_val_type) + to_str(right_val, right_val_type), "string")
            if left_val_type == right_val_type == "int":
                return Variable(left_val + right_val, "int")
            raise Exception(f"Operação '+' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "MINUS":
            if left_val_type == right_val_type == "int":
                return Variable(left_val - right_val, "int")
            raise Exception(f"Operação '-' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "MULT":
            if left_val_type == right_val_type == "int":
                return Variable(left_val * right_val, "int")
            raise Exception(f"Operação '*' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "DIV":
            if left_val_type == right_val_type == "int":
                if right_val == 0:
                    raise Exception("Divisão por zero.")
                return Variable(left_val // right_val, "int")
            raise Exception(f"Operação '/' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "EQUAL":
            if left_val_type != right_val_type:
                raise Exception(f"Operação '==' inválida entre '{left_val_type}' e '{right_val_type}'")
            return Variable(1 if left_val == right_val else 0, "bool")

        elif self.value == "LT":
            if left_val_type == right_val_type == "int":
                return Variable(1 if left_val < right_val else 0, "bool")
            if left_val_type == right_val_type == "string":
                return Variable(1 if str(left_val) < str(right_val) else 0, "bool")
            raise Exception(f"Operação '<' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "GT":
            if left_val_type == right_val_type == "int":
                return Variable(1 if left_val > right_val else 0, "bool")
            if left_val_type == right_val_type == "string":
                return Variable(1 if str(left_val) > str(right_val) else 0, "bool")
            raise Exception(f"Operação '>' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "AND":
            if left_val_type == right_val_type == "bool":
                return Variable(1 if (left_val != 0 and right_val != 0) else 0, "bool")
            raise Exception(f"Operação '&&' inválida entre '{left_val_type}' e '{right_val_type}'")

        elif self.value == "OR":
            if left_val_type == right_val_type == "bool":
                return Variable(1 if (left_val != 0 or right_val != 0) else 0, "bool")
            raise Exception(f"Operação '||' inválida entre '{left_val_type}' e '{right_val_type}'")

        else:
            raise Exception(f"Operador binário desconhecido: {self.value}")


class UnOp(Node):
    def evaluate(self, st: SymbolTable):
        val = self.children[0].evaluate(st)
        if self.value == "PLUS":
            return Variable(val.value, "int")
        elif self.value == "MINUS":
            return Variable(-val.value, "int")
        elif self.value == "NOT":
            return Variable(0 if val.value else 1, "bool")
        else:
            raise Exception(f"Operador unário desconhecido: {self.value}")


class IntVal(Node):
    def evaluate(self, st: SymbolTable):
        return Variable(self.value, "int")


class NoOp(Node):
    def evaluate(self, st):
        pass


class Assignment(Node):
    def evaluate(self, st: SymbolTable):
        var_name = self.children[0].value
        var_value = self.children[1].evaluate(st)
        st.set(var_name, var_value.value, var_value.type)


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
    def evaluate(self, st: SymbolTable):
        var_name = self.children[0].value
        dec_type = self.value  # "int" | "string" | "bool"
        if dec_type not in ("int", "string", "bool"):
            raise Exception(f"Tipo de variável desconhecido: {dec_type}")

        defaults = {"int": 0, "string": "", "bool": 0}

        if len(self.children) == 2:
            rhs = self.children[1].evaluate(st)
            if rhs.type != dec_type:
                raise Exception(
                    f"Tipo incompatível na inicialização da variável '{var_name}'. "
                    f"Esperado '{dec_type}', recebido '{rhs.type}'."
                )
            st.create_variable(var_name, Variable(rhs.value, dec_type))
            return Variable(rhs.value, dec_type)

        dv = defaults[dec_type]
        st.create_variable(var_name, Variable(dv, dec_type))
        return Variable(dv, dec_type)


class Print(Node):
    def evaluate(self, st: SymbolTable):
        value = self.children[0].evaluate(st)
        if value.type == "bool":
            print("true" if value.value != 0 else "false")
        else:
            print(value.value)


class Read(Node):
    def evaluate(self, st: SymbolTable):
        value = int(input())
        return Variable(value, "int")


# ===== Controle de fluxo com 'return' via exceção =====

class ReturnException(Exception):
    def __init__(self, value: Variable):
        super().__init__("Return")
        self.value = value


class Return(Node):
    # value = "RETURN", children = [expr]
    def evaluate(self, st: SymbolTable):
        val = self.children[0].evaluate(st)
        raise ReturnException(val)


class Block(Node):
    def evaluate(self, st: SymbolTable):
        for stmt in self.children:
            if isinstance(stmt, Block):
                # novo escopo para bloco filho
                child_st = SymbolTable({}, parent=st)
                try:
                    stmt.evaluate(child_st)
                except ReturnException as e:
                    # propaga para cima
                    raise e
            else:
                try:
                    stmt.evaluate(st)
                except ReturnException as e:
                    raise e


class While(Node):
    def evaluate(self, st: SymbolTable):
        cond = self.children[0].evaluate(st)
        if cond.type != "bool":
            raise Exception("Condição do 'while' deve ser do tipo bool.")
        while self.children[0].evaluate(st).value != 0:
            try:
                self.children[1].evaluate(st)
            except ReturnException as e:
                # return dentro do while propaga
                raise e


class If(Node):
    def evaluate(self, st: SymbolTable):
        cond = self.children[0].evaluate(st)
        if cond.type != "bool":
            raise Exception("Condição do 'if' deve ser do tipo bool.")
        try:
            if cond.value != 0:
                self.children[1].evaluate(st)
            elif len(self.children) == 3:
                self.children[2].evaluate(st)
        except ReturnException as e:
            raise e


# ===== Funções =====

class FuncDec(Node):
    """
    value = return_type ('int'|'string'|'bool'|'void')
    children = [ Identifier(nome), VarDec(param1), ..., VarDec(paramN), Block(corpo) ]
    """
    def evaluate(self, st: SymbolTable):
        func_name = self.children[0].value
        st.create_variable(func_name, Variable(self, self.value, is_func=True))
        return Variable(None, "void")


class FuncCall(Node):
    """
    value = nome da função
    children = [ exprArg1, exprArg2, ... ]
    """
    def evaluate(self, st: SymbolTable):
        sym = st.getter(self.value)
        if not sym.is_func:
            raise Exception(f"'{self.value}' não é função.")
        func_node: FuncDec = sym.value

        params = func_node.children[1:-1]  # VarDecs
        body = func_node.children[-1]      # Block

        if len(self.children) != len(params):
            raise Exception(f"Chamada incorreta: função '{self.value}' espera {len(params)} argumento(s), recebeu {len(self.children)}.")

        call_st = SymbolTable({}, parent=st)

        # parâmetros
        for arg_expr, param_vardec in zip(self.children, params):
            param_name = param_vardec.children[0].value
            param_type = param_vardec.value  # "int"|"string"|"bool"

            # declarar no escopo da chamada
            default = 0 if param_type == "int" else ("" if param_type == "string" else 0)
            call_st.create_variable(param_name, Variable(default, param_type))

            # avaliar o argumento no escopo de chamada do *caller*
            arg_val = arg_expr.evaluate(st)
            if arg_val.type != param_type:
                raise Exception(f"Tipo do argumento '{param_name}' inválido: esperado {param_type}, recebeu {arg_val.type}")
            call_st.set(param_name, arg_val.value, arg_val.type)

        # executar corpo
        ret_value = None
        try:
            body.evaluate(call_st)
        except ReturnException as e:
            ret_value = e.value  # Variable

        # checar tipo de retorno
        ret_type = func_node.value  # "int"|"string"|"bool"|"void"
        if ret_type == "void":
            # se houver 'return expr' numa void, você pode transformar em erro;
            # aqui ignoramos o ret_value se aparecer.
            return Variable(None, "void")
        else:
            if not isinstance(ret_value, Variable):
                raise Exception(f"Função '{self.value}' deve retornar '{ret_type}', mas não retornou.")
            if ret_value.type != ret_type:
                raise Exception(f"Função '{self.value}' retornou tipo '{ret_value.type}', esperado '{ret_type}'.")
            return ret_value


# =============================
# Parser
# =============================

class Parser:
    lex: Lexer = None

    @staticmethod
    def parse_program() -> Block:
        """
        Programa = { DeclFunc | DeclVar | Statement }*  EOF
        (nós de função e var podem aparecer misturados; o run chamará main())
        """
        statements = []
        while Parser.lex.next.kind != "EOF":
            if Parser.lex.next.kind == "FUNC":
                statements.append(Parser.parse_func_declaration())
            else:
                statements.append(Parser.parse_statement())
        return Block("BLOCK", statements)

    @staticmethod
    def parse_block() -> Block:
        if Parser.lex.next.kind == "OPEN_BRA":
            Parser.lex.select_next()
            children = []
            while Parser.lex.next.kind != "CLOSE_BRA":
                node = Parser.parse_statement()
                children.append(node)
            if Parser.lex.next.kind == "CLOSE_BRA":
                if len(children) == 0:
                    raise Exception("Erro: bloco vazio não é permitido")
                Parser.lex.select_next()
                return Block("BLOCK", children)
            else:
                raise Exception("Erro: esperado '}' no final do bloco")
        else:
            raise Exception("Erro: esperado '{' no início do bloco")

    @staticmethod
    def parse_func_declaration() -> Node:
        # estamos em FUNC
        Parser.lex.select_next()
        if Parser.lex.next.kind != "IDEN":
            raise Exception("Esperado identificador após 'func'.")
        func_name = Identifier(Parser.lex.next.value, [])
        Parser.lex.select_next()

        if Parser.lex.next.kind != "OPEN_PAR":
            raise Exception("Esperado '(' na declaração de função.")
        Parser.lex.select_next()

        params = []
        if Parser.lex.next.kind != "CLOSE_PAR":
            while True:
                if Parser.lex.next.kind != "IDEN":
                    raise Exception("Esperado identificador de parâmetro.")
                p_id = Identifier(Parser.lex.next.value, [])
                Parser.lex.select_next()
                if Parser.lex.next.kind != "TYPE":
                    raise Exception("Esperado tipo do parâmetro.")
                p_type = Parser.lex.next.value
                Parser.lex.select_next()
                params.append(VarDec(p_type, [p_id]))
                if Parser.lex.next.kind == "COMMA":
                    Parser.lex.select_next()
                    continue
                break
        if Parser.lex.next.kind != "CLOSE_PAR":
            raise Exception("Esperado ')' após parâmetros.")
        Parser.lex.select_next()

        # tipo de retorno opcional
        ret_type = "void"
        if Parser.lex.next.kind == "TYPE":
            ret_type = Parser.lex.next.value
            Parser.lex.select_next()

        # corpo
        body = Parser.parse_block()
        return FuncDec(ret_type, [func_name] + params + [body])

    @staticmethod
    def parse_statement() -> Node:
        if Parser.lex.next.kind == "END":
            folha = NoOp("NoOp", [])
            Parser.lex.select_next()
            return folha

        elif Parser.lex.next.kind == "RETURN":
            Parser.lex.select_next()
            expr = Parser.parseBoolExpression()
            if Parser.lex.next.kind != "END":
                raise Exception("Esperado fim de linha após 'return'.")
            Parser.lex.select_next()
            return Return("RETURN", [expr])

        elif Parser.lex.next.kind == "IDEN":
            id_node = Identifier(Parser.lex.next.value, [])
            Parser.lex.select_next()
            # chamada de função como statement?
            if Parser.lex.next.kind == "OPEN_PAR":
                Parser.lex.select_next()
                args = []
                if Parser.lex.next.kind != "CLOSE_PAR":
                    while True:
                        args.append(Parser.parseBoolExpression())
                        if Parser.lex.next.kind == "COMMA":
                            Parser.lex.select_next()
                            continue
                        break
                if Parser.lex.next.kind != "CLOSE_PAR":
                    raise Exception("Esperado ')' na chamada de função.")
                Parser.lex.select_next()
                if Parser.lex.next.kind != "END":
                    raise Exception("Esperado fim de linha após chamada de função.")
                Parser.lex.select_next()
                return FuncCall(id_node.value, args)

            # atribuição
            if Parser.lex.next.kind != "ASSIGN":
                raise Exception("Esperado '=' ou '(' após identificador.")
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
            then_node = Parser.parse_statement()
            if Parser.lex.next.kind == "ELSE":
                Parser.lex.select_next()
                if Parser.lex.next.kind == "END":
                    raise Exception("Unexpected token NEWLINE")
                else_node = Parser.parse_statement()
                return If("IF", [cond_node, then_node, else_node])
            return If("IF", [cond_node, then_node])

        elif Parser.lex.next.kind == "VAR":
            Parser.lex.select_next()
            if Parser.lex.next.kind != "IDEN":
                raise Exception("Esperado identificador após 'var'.")
            id_node = Identifier(Parser.lex.next.value, [])
            Parser.lex.select_next()
            if Parser.lex.next.kind != "TYPE":
                raise Exception("Esperado tipo após identificador.")
            type_node = Parser.lex.next.value
            Parser.lex.select_next()
            expr_node = None
            if Parser.lex.next.kind == "ASSIGN":
                Parser.lex.select_next()
                expr_node = Parser.parseBoolExpression()
            if Parser.lex.next.kind != "END":
                raise Exception("Esperado fim de linha após declaração.")
            Parser.lex.select_next()
            return VarDec(type_node, [id_node] + ([expr_node] if expr_node else []))

        elif Parser.lex.next.kind == "WHILE":
            Parser.lex.select_next()
            cond_node = Parser.parseBoolExpression()
            if Parser.lex.next.kind == "END":
                raise Exception("Unexpected token NEWLINE")
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

        elif Parser.lex.next.kind == "FUNC":
            # permite função fora de parse_program (robustez)
            return Parser.parse_func_declaration()

        else:
            raise Exception(f"Token inesperado: {Parser.lex.next.kind}")

    @staticmethod
    def parseBoolExpression():
        node = Parser.parseBoolTerm()
        if Parser.lex.next.kind == "OR":
            Parser.lex.select_next()
            node = BinOp("OR", [node, Parser.parseBoolTerm()])
        return node

    @staticmethod
    def parseBoolTerm():
        node = Parser.parser_relExpression()
        if Parser.lex.next.kind == "AND":
            Parser.lex.select_next()
            node = BinOp("AND", [node, Parser.parser_relExpression()])
        return node

    @staticmethod
    def parser_relExpression():
        node = Parser.parse_expression()
        if Parser.lex.next.kind in ("EQUAL", "LT", "GT"):
            op = Parser.lex.next.kind
            Parser.lex.select_next()
            node = BinOp(op, [node, Parser.parse_expression()])
        return node

    @staticmethod
    def parse_expression():
        node = Parser.parse_term()
        while Parser.lex.next.kind in ("PLUS", "MINUS"):
            if Parser.lex.next.kind == "PLUS":
                Parser.lex.select_next()
                node = BinOp("PLUS", [node, Parser.parse_term()])
            elif Parser.lex.next.kind == "MINUS":
                Parser.lex.select_next()
                node = BinOp("MINUS", [node, Parser.parse_term()])
        return node

    @staticmethod
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

    @staticmethod
    def parse_factor():
        if Parser.lex.next.kind == "INT":
            node = IntVal(Parser.lex.next.value, [])
            Parser.lex.select_next()
            return node

        elif Parser.lex.next.kind == "OPEN_PAR":
            Parser.lex.select_next()
            node = Parser.parseBoolExpression()
            if Parser.lex.next.kind != "CLOSE_PAR":
                raise Exception("Faltando )")
            Parser.lex.select_next()
            return node

        elif Parser.lex.next.kind == "BOOL":
            node = BoolVal(Parser.lex.next.value, [])
            Parser.lex.select_next()
            return node

        elif Parser.lex.next.kind == "STR":
            node = StringVal(Parser.lex.next.value, [])
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
            id_name = Parser.lex.next.value
            Parser.lex.select_next()
            # chamada de função como expressão?
            if Parser.lex.next.kind == "OPEN_PAR":
                Parser.lex.select_next()
                args = []
                if Parser.lex.next.kind != "CLOSE_PAR":
                    while True:
                        args.append(Parser.parseBoolExpression())
                        if Parser.lex.next.kind == "COMMA":
                            Parser.lex.select_next()
                            continue
                        break
                if Parser.lex.next.kind != "CLOSE_PAR":
                    raise Exception("Esperado ')' na chamada de função.")
                Parser.lex.select_next()
                return FuncCall(id_name, args)
            return Identifier(id_name, [])

        else:
            raise Exception(f"Erro: token inesperado '{Parser.lex.next.kind}'")

    @staticmethod
    def run(code: str) -> Node:
        clean = Prepro.filter(code)
        Parser.lex = Lexer(clean, 0, None)
        Parser.lex.select_next()
        ast = Parser.parse_program()
        if Parser.lex.next.kind != "EOF":
            raise Exception("Tokens remanescentes após o fim do programa.")
        return ast


# =============================
# Entrypoint
# =============================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 main.py teste.go")
        sys.exit(1)

    filename = sys.argv[1]
    with open(filename, "r", encoding="utf-8") as f:
        code = f.read()

    root = Parser.run(code)
    global_st = SymbolTable({})

    # 1) avalia topo: registra funções, inicializa globais
    root.evaluate(global_st)

    # 2) chama main() automaticamente (sem argumentos)
    try:
        call_main = FuncCall("main", [])
        ret = call_main.evaluate(global_st)  # pode retornar Variable ou 'void'
        # se quiser exibir o retorno quando for int/string/bool, pode descomentar:
        # if isinstance(ret, Variable) and ret.type != "void":
        #     print(ret.value)
    except Exception as e:
        raise
