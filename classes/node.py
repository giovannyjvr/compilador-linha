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


class Block:
    def evaluate(self, st: SymbolTable):
        for stmt in self.children:
            stmt.evaluate(st)

class Print:
    def evaluate(self, st: SymbolTable):
        value = self.children[0].evaluate(st)
        print(value)