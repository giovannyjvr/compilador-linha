def main(string):
    soma = 0
    numero = ""
    space = False
    token = ""
    operacao = "+"

    for s in string:
        if s.isdigit() :
            if space == True and token == "n":
                return ValueError("Invalid expression")
            numero += s
            token = "n"
            space = False

        elif s == " ":
            space = True

        elif s in "+-":
            if token != "s":
                return ValueError("Invalid expression")
            
            if token == "" and s == "-":
                numero = ""
                operacao = s
                token = "s"
                space = False
            
            elif token == "" and s == "+":
                return ValueError("Invalid expression")
            
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
            try:
                raise ValueError("Invalid character found")
            except ValueError as e:
                return(e)

    if token =="s":
        return ValueError("Invalid expression")
    if operacao == "+":
        soma += int(numero)
    elif operacao == "-":
        soma -= int(numero)
        
    return soma

