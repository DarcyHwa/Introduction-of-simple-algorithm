class Casilla():
    def __init__(self, f, c):
        self.fila=f
        self.col=c

    # develve la coordenada de fila.
    # En caso igual a -1 es que no es una coordenada valida
    def getFila (self):
        return self.fila

    # develve la coordenada de columna
    # En caso igual a -1 es que no es una coordenada valida
    def getCol (self):
        return self.col
