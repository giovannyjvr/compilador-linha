import sys

string = sys.argv[1]

soma = 0
numero = ""
space = False
token = ""
operacao = "+"

for s in string:
    if s.isdigit() :
        if space == True and token == "n":
            raise Exception("Invalid expression 'num  space  num'")
        numero += s
        token = "n"
        space = False

    elif s == " ":
        space = True

    elif s in "+-":
        if token == "s":
            raise Exception("Invalid expression")

        elif token == "":
            raise Exception("Invalid expression")

        else:
            if operacao == "+":
                soma += int(numero)
            elif operacao == "-":
                soma -= int(numero)
            numero = ""
            operacao = s
            token = "s"
            space = False

    else:
        raise Exception("Invalid character found")

if token =="s":
    raise Exception("Invalid expression")
if operacao == "+":
    soma += int(numero)
elif operacao == "-":
    soma -= int(numero)
    
print(soma)
