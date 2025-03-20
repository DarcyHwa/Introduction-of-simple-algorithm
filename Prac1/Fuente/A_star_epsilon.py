from A_star import A_star, CONST_DIRECCION_DE_MOVIMIENTO, Node

class A_star_epsilon(A_star):

    def __init__(self, conejo_x, conejo_y, zanahoria_x, zanahoria_y, mapa_file, epsilon=0.5):
        super().__init__(conejo_x, conejo_y, zanahoria_x, zanahoria_y, mapa_file)
        self.epsilon = epsilon

    def get_calories_focal(self, x, y):
        """
        Segunda heurística basada en calorías para la lista focal
        """
        return self.calc_calorias(x, y)

    def get_lista_focal(self, lista_abierta, f_min):
        """
        Obtiene la lista focal basada en el factor épsilon
        """
        return [nodo for nodo in lista_abierta if nodo.f <= (1 + self.epsilon) * f_min]

    def get_camino(self):
        """
        Algoritmo A* epsilon
        """
        lista_abierta = []
        lista_cerrada = []
        nodo_inicial = Node(self.conejo_x, self.conejo_y)
        lista_abierta.append(nodo_inicial)

        while lista_abierta:
            # Encontrar el mejor valor f en la lista abierta
            f_min = min(nodo.f for nodo in lista_abierta)

            lista_focal = self.get_lista_focal(lista_abierta, f_min)

            # Seleccionar el nodo de la lista focal con menor valor de heurística focal
            nodo_actual = min(lista_focal,
                              key=lambda x: self.get_calories_focal(x.x, x.y))

            lista_abierta.remove(nodo_actual)
            lista_cerrada.append(nodo_actual)

            # Si llegamos al destino, reconstruimos el camino
            if nodo_actual.x == self.zanahoria_x and nodo_actual.y == self.zanahoria_y:
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
                                                      y)  # Calcula la heurística h (distancia Manhattan o Euclidiana o Chebyshev)
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