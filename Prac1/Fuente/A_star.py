import math
import time

from mapa import Mapa

######################
# Tipos de terreno
Roca = 5
Agua = 4
Hierba = 0
######################
COST_CAL_HIERBA = 2  # Calorías por paso en hierba
COST_CAL_AGUA = 4  # Calorías por paso en agua, color azul
COST_CAL_ROCA = 6  # Calorías por paso en roca, color marron
CONST_COSTE_MOVIMIENTO_HORIZONTAL_VERTICAL = 1
CONST_COSTE_MOVIMIENTO_DIAGONAL = 1.5
######################

CONST_DIRECCION_DE_MOVIMIENTO = [
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
]  # Movimiento horizontal, vertical y diagonal


class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.g = 0
        self.h = 0
        self.f = 0
        self.parent = parent

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def read_mapa(mapas) -> Mapa:
    """
    Lee el archivo de mapa y devuelve un objeto Mapa.
    """
    mapa = Mapa(mapas)
    print(mapa)
    return mapa


class A_star:
    """
    Clase principal para el algoritmo A*.
    """
    def __init__(self, conejo_x, conejo_y, zanahoria_x, zanahoria_y, mapa_file) -> None:
        self.nodos_visitados = []
        self.camino = None
        self.conejo_x = conejo_x
        self.conejo_y = conejo_y
        self.zanahoria_x = zanahoria_x
        self.zanahoria_y = zanahoria_y
        self.mapa = read_mapa(mapa_file)
        self.cost_cal = 0  # Coste de calorías total
        self.cost_mov = -1  # Coste de movimiento total

    def is_valid_map_coordenate(self, x, y) -> bool:
        """
        Comprueba si la coordenada es válida en el mapa.
        """
        celda = self.mapa.mapa[y][x]

        return celda != 1  # No es transitable si es una celda bloqueada (1)

    def calc_calorias(self, x, y):
        """
        Calcula las calorías según el tipo de terreno (hierba, agua, roca).
        """
        celda = self.mapa.getCelda(y, x)
        if celda == Hierba:  # Hierba
            return COST_CAL_HIERBA
        elif celda == Agua:  # Agua
            return COST_CAL_AGUA
        elif celda == Roca:  # Roca
            return COST_CAL_ROCA
        return 0  # Si es un terreno desconocido o transitable sin calorías adicionales

    def manhattan(self, x, y) -> int:
        """
        Función heurística: distancia Manhattan al objetivo.
        """
        return abs(x - self.zanahoria_x) + abs(y - self.zanahoria_y)

    def euclidean(self, x, y) -> float:
        """
        Función heurística: distancia Euclidiana al objetivo.
        """
        return math.sqrt((x - self.zanahoria_x) ** 2 + (y - self.zanahoria_y) ** 2)

    def chebyshev(self, x, y) -> int:
        """
        Función heurística: distancia Chebyshev al objetivo.
        """
        return max(abs(x - self.zanahoria_x), abs(y - self.zanahoria_y))

    def octile(self, x, y) -> int:
        """
        Función heurística: distancia octile al objetivo.
        """
        F = math.sqrt(2) - 1
        if(abs(x - self.zanahoria_x) < abs(y - self.zanahoria_y)):
            return F * abs(x - self.zanahoria_x) + abs(y - self.zanahoria_y)
        else:
            return F * abs(y - self.zanahoria_y) + abs(x - self.zanahoria_x)

    def diijkstra(self, x, y) -> int:
        """
        Función heurística: distancia diijkstra al objetivo. h = 0
        """
        return 0
    @staticmethod
    def f(g, h) -> int:
        """
        Función de evaluación f(n) = g(n) + h(n)
        """
        return g + h

    def is_valid_position(self, x, y) -> bool:
        """
        Comprueba si la posición es válida en el mapa, que no este en un posicion como roca, agua o hierror.
        """
        return 0 <= x < self.mapa.getAncho() and 0 <= y < self.mapa.getAlto()

    def reconstruir_camino(self, nodo_final):
        """
        Reconstruye el camino desde el nodo final hasta el inicial
        """
        camino = []
        n = nodo_final
        while n:
            camino.append((n.x, n.y))
            n = n.parent
        ##############################################
        # Eliminar la posición inicial y final para no pintar sobre ellos
        # camino.pop(0)
        camino.pop(-1)  # inicial
        ##############################################
        for c in camino:
            self.cost_cal += self.calc_calorias(c[0], c[1])
        return camino

    def calc_coste_movimiento(self, dx, dy):
        """
        Calcula el coste de movimiento según el tipo de movimiento
        Movimiento diagonal: coste de 1.5
        Movimiento horizontal y vertical: 1
        """
        if dx == 1 and dy == 1:
            return CONST_COSTE_MOVIMIENTO_DIAGONAL  # Movimiento diagonal
        else:
            return CONST_COSTE_MOVIMIENTO_HORIZONTAL_VERTICAL  # Movimiento horizontal o vertical

    def get_camino(self):
        """Funcion principal para calcular el camino optimo desde inicio hasta destino(conejo y zanahoria)"""
        lista_abierta = []
        lista_cerrada = []
        nodo_inicial = Node(self.conejo_x, self.conejo_y)
        lista_abierta.append(nodo_inicial)

        while lista_abierta:
            # Seleccionamos el nodo con el menor valor de f
            nodo_actual = min(lista_abierta, key=lambda x: x.f)
            lista_abierta.remove(nodo_actual)
            lista_cerrada.append(nodo_actual)
            # Si llegamos al destino, reconstruimos el camino
            if nodo_actual.x == self.zanahoria_x and nodo_actual.y == self.zanahoria_y:
                # print("Coste de movimiento: ", nodo_actual.g)
                self.cost_mov = nodo_actual.g
                return self.reconstruir_camino(nodo_actual)

            # Expandimos los nodos vecinos
            for direccion in CONST_DIRECCION_DE_MOVIMIENTO:
                x = nodo_actual.x + direccion[0]
                y = nodo_actual.y + direccion[1]

                if self.is_valid_position(x, y) and self.is_valid_map_coordenate(x, y):
                    nodo_siguiente = Node(x, y, nodo_actual)

                    if nodo_siguiente in lista_cerrada:
                        continue

                    # Actualizar el coste g basado en el tipo de movimiento
                    nodo_siguiente.g = (
                                nodo_actual.g + self.calc_coste_movimiento(abs(direccion[0]), abs(direccion[1])))
                    nodo_siguiente.h = self.octile(x,
                                                      y)  # Calcula la heurística h (distancia Manhattan, Euclidiana, diijkstra, octile)
                    nodo_siguiente.f = self.f(nodo_siguiente.g, nodo_siguiente.h)

                    print(f"({x} {y})", "g: ", nodo_siguiente.g, "h: ", nodo_siguiente.h, "f: ", nodo_siguiente.f)
                    self.nodos_visitados.append((x, y))
                    # Si el nodo no está en lista abierta, lo añadimos
                    if nodo_siguiente not in lista_abierta:
                        lista_abierta.append(nodo_siguiente)
                    else:
                        # Si ya está en la lista abierta, comprobamos si este camino es mejor
                        existing_node = next(n for n in lista_abierta if n == nodo_siguiente)
                        if nodo_siguiente.g < existing_node.g:
                            existing_node.g = nodo_siguiente.g
                            existing_node.f = nodo_siguiente.f
                            existing_node.parent = nodo_actual

        return None  # No se encontró solución

    def get_calorias(self) -> int:
        """
        Obtener los calorias consumidas en total durante el camino
        """
        return self.cost_cal

    def get_movimiento(self) -> int:
        """
        Obtiene el coste de movimiento total.
        """
        return self.cost_mov

    def draw_map(self, mapi) -> list:
        """
        Pintar el camino en el mapa.

        Devuelve el mapa con el camino pintado.
        """
        if self.camino is None:
            return mapi
        for y in range(self.mapa.getAlto()):
            for x in range(self.mapa.getAncho()):
                if (x, y) in self.camino:
                    mapi[y][x] = "X"
        return mapi

    def main(self, camino):
        """
        Función principal para ejecutar el algoritmo A*.
        """
        print(self.mapa)
        start_time = time.perf_counter()

        self.camino = self.get_camino()
        self.draw_map(camino)

        print("Tiempo de ejecución: ", time.perf_counter() - start_time)
        print("Coste de movimiento: ", self.cost_mov)
        print("Coste de calorías: ", self.cost_cal)
        # print("Nodos visitados: ", self.nodos_visitados)
        print("numero de nodos visitados: ", len(self.nodos_visitados))

    def getNumNodes(self):
        return len(self.nodos_visitados)