import logging
import sys, pygame
from casilla import *
from mapa import *
from pygame.locals import *

from A_star import A_star
from A_star_epsilon import A_star_epsilon

MARGEN = 5
MARGEN_INFERIOR = 60
TAM = 30
NEGRO = (0, 0, 0)
HIERBA = (250, 180, 160)  # hierba color carne
MURO = (30, 70, 140)  # azul oscuro
AGUA = (173, 216, 230)  # azul
ROCA = (110, 75, 48)  # marron
AMARILLO = (255,255,0)  # amarillo, camino

# ---------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------


# Devuelve si una casilla del mapa se puede seleccionar como destino o como origen
def bueno(mapi, pos):
    res = False

    if (
        mapi.getCelda(pos.getFila(), pos.getCol()) == 0
        or mapi.getCelda(pos.getFila(), pos.getCol()) == 4
        or mapi.getCelda(pos.getFila(), pos.getCol()) == 5
    ):
        res = True

    return res


# Devuelve si una posición de la ventana corresponde al mapa
def esMapa(mapi, posicion):
    res = False

    if (
        posicion[0] > MARGEN
        and posicion[0] < mapi.getAncho() * (TAM + MARGEN) + MARGEN
        and posicion[1] > MARGEN
        and posicion[1] < mapi.getAlto() * (TAM + MARGEN) + MARGEN
    ):
        res = True

    return res


# Devuelve si se ha pulsado algún botón
# -1 en caso que no ha pulsado un boton
def pulsaBoton(mapi, posicion):
    res = -1

    if (
        posicion[0] > (mapi.getAncho() * (TAM + MARGEN) + MARGEN) // 2 - 65
        and posicion[0] < (mapi.getAncho() * (TAM + MARGEN) + MARGEN) // 2 - 15
        and posicion[1] > mapi.getAlto() * (TAM + MARGEN) + MARGEN + 10
        and posicion[1] < MARGEN_INFERIOR + mapi.getAlto() * (TAM + MARGEN) + MARGEN
    ):
        res = 1
    elif (
        posicion[0] > (mapi.getAncho() * (TAM + MARGEN) + MARGEN) // 2 + 15
        and posicion[0] < (mapi.getAncho() * (TAM + MARGEN) + MARGEN) // 2 + 65
        and posicion[1] > mapi.getAlto() * (TAM + MARGEN) + MARGEN + 10
        and posicion[1] < MARGEN_INFERIOR + mapi.getAlto() * (TAM + MARGEN) + MARGEN
    ):
        res = 2

    return res


# Construye la matriz para guardar el camino
def inic(mapi):
    cam = []
    for i in range(mapi.alto):
        cam.append([])
        for j in range(mapi.ancho):
            cam[i].append(".")

    return cam


# función principal
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    Logger = logging.getLogger("Practica1")
    pygame.init()

    reloj = pygame.time.Clock()

    if len(sys.argv) == 1:  # si no se indica un mapa coge mapa0.txt por defecto
        file = r"./Mundos/mapa5.txt"
    else:
        file = sys.argv[-1]

    mapi = Mapa(file)
    camino = inic(mapi)

    anchoVentana = mapi.getAncho() * (TAM + MARGEN) + MARGEN
    altoVentana = MARGEN_INFERIOR + mapi.getAlto() * (TAM + MARGEN) + MARGEN
    dimension = [anchoVentana, altoVentana]
    screen = pygame.display.set_mode(dimension)
    pygame.display.set_caption("Practica 1")

    boton1 = pygame.image.load(r"boton1.png").convert()
    boton1 = pygame.transform.scale(boton1, [50, 30])

    boton2 = pygame.image.load(r"boton2.png").convert()
    boton2 = pygame.transform.scale(boton2, [50, 30])

    personaje = pygame.image.load(r"rabbit.png").convert()
    personaje = pygame.transform.scale(personaje, [TAM, TAM])

    objetivo = pygame.image.load(r"carrot.png").convert()
    objetivo = pygame.transform.scale(objetivo, [TAM, TAM])

    coste = -1
    cal = 0
    running = True
    origen = Casilla(-1, -1)
    destino = Casilla(-1, -1)
    numNodos = 0
    while running:
        # procesamiento de eventos
        for event in pygame.event.get():

            # 退出事件
            if event.type == pygame.QUIT:
                running = False

            # 鼠标按下事件
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                if pulsaBoton(mapi, pos) == 1 or pulsaBoton(mapi, pos) == 2:
                    if origen.getFila() == -1 or destino.getFila() == -1:
                        print("Error: No hay origen o destino")
                    else:

                        camino = inic(mapi)

                        # buton de A*
                        if pulsaBoton(mapi, pos) == 1:
                            print("-" * 50)
                            Logger.info("A*")

                            a_star = A_star(origen.getCol(), origen.getFila(),
                                            destino.getCol(), destino.getFila(),
                                            file)
                            a_star.main(camino)
                            # a_star.draw_map(camino)
                            coste = a_star.get_movimiento()
                            cal = a_star.get_calorias()
                            numNodos = a_star.getNumNodes()
                            if coste == -1:
                                print("No hay camino posible")
                        # buton de A_e
                        else:
                            Logger.info("A* subepsilon")
                            a_start_e = A_star_epsilon(origen.getCol(), origen.getFila(),
                                                       destino.getCol(), destino.getFila(), file, epsilon=0.5)
                            a_start_e.main(camino)
                            coste = a_start_e.get_movimiento()
                            cal = a_start_e.get_calorias()
                            numNodos = a_start_e.getNumNodes()
                            if coste == -1:
                                print("No hay camino posible")
                elif esMapa(mapi, pos):
                    if event.button == 1:  # botón izquierdo de raton
                        colOrigen = pos[0] // (TAM + MARGEN)
                        filOrigen = pos[1] // (TAM + MARGEN)
                        casO = Casilla(filOrigen, colOrigen)
                        if bueno(mapi, casO):
                            origen = casO
                        else:  # se ha hecho click en una celda no accesible
                            print("Error: Esa casilla no es válida")
                    elif event.button == 3:  # botón derecho de raton
                        colDestino = pos[0] // (TAM + MARGEN)
                        filDestino = pos[1] // (TAM + MARGEN)
                        casD = Casilla(filDestino, colDestino)
                        if bueno(mapi, casD):
                            destino = casD
                        else:  # se ha hecho click en una celda no accesible
                            print("Error: Esa casilla no es válida")

        # código de dibujo
        # limpiar pantalla
        screen.fill(NEGRO)
        # pinta mapa
        for fil in range(mapi.getAlto()):
            for col in range(mapi.getAncho()):
                if (
                    camino[fil][col] != "."
                ):  # casillas que se puede poner rabbit y carol
                    pygame.draw.rect(
                        screen,
                        AMARILLO,
                        [
                            (TAM + MARGEN) * col + MARGEN,
                            (TAM + MARGEN) * fil + MARGEN,
                            TAM,
                            TAM,
                        ],
                        0,
                    )
                elif mapi.getCelda(fil, col) == 0:
                    pygame.draw.rect(
                        screen,
                        HIERBA,
                        [
                            (TAM + MARGEN) * col + MARGEN,
                            (TAM + MARGEN) * fil + MARGEN,
                            TAM,
                            TAM,
                        ],
                        0,
                    )
                elif mapi.getCelda(fil, col) == 1:
                    pygame.draw.rect(
                        screen,
                        MURO,
                        [
                            (TAM + MARGEN) * col + MARGEN,
                            (TAM + MARGEN) * fil + MARGEN,
                            TAM,
                            TAM,
                        ],
                        0,
                    )
                elif mapi.getCelda(fil, col) == 4:
                    pygame.draw.rect(
                        screen,
                        AGUA,
                        [
                            (TAM + MARGEN) * col + MARGEN,
                            (TAM + MARGEN) * fil + MARGEN,
                            TAM,
                            TAM,
                        ],
                        0,
                    )
                elif mapi.getCelda(fil, col) == 5:
                    pygame.draw.rect(
                        screen,
                        ROCA,
                        [
                            (TAM + MARGEN) * col + MARGEN,
                            (TAM + MARGEN) * fil + MARGEN,
                            TAM,
                            TAM,
                        ],
                        0,
                    )

        # pinta origen
        screen.blit(
            personaje,
            [
                (TAM + MARGEN) * origen.getCol() + MARGEN,
                (TAM + MARGEN) * origen.getFila() + MARGEN,
            ],
        )
        # pinta destino
        screen.blit(
            objetivo,
            [
                (TAM + MARGEN) * destino.getCol() + MARGEN,
                (TAM + MARGEN) * destino.getFila() + MARGEN,
            ],
        )
        # pinta botón
        screen.blit(
            boton1,
            [anchoVentana // 2 - 65, mapi.getAlto() * (TAM + MARGEN) + MARGEN + 10],
        )
        screen.blit(
            boton2,
            [anchoVentana // 2 + 15, mapi.getAlto() * (TAM + MARGEN) + MARGEN + 10],
        )
        # pinta coste y energía
        if coste != -1:
            fuente = pygame.font.Font(None, 25)
            textoCoste = fuente.render("Coste: " + str(coste), True, AMARILLO)
            screen.blit(
                textoCoste,
                [anchoVentana - 90, mapi.getAlto() * (TAM + MARGEN) + MARGEN + 15],
            )
            textoEnergía = fuente.render("Cal: " + str(cal), True, AMARILLO)
            screen.blit(
                textoEnergía, [5, mapi.getAlto() * (TAM + MARGEN) + MARGEN + 15]
            )
            textoNumeroTotaldeNodos = fuente.render("Nodos visitados: " + str(numNodos), True, AMARILLO)
            screen.blit(
                textoNumeroTotaldeNodos, [5, mapi.getAlto() * (TAM + MARGEN) + MARGEN + 35]
            )

        # actualizar pantalla
        pygame.display.flip()
        reloj.tick(40)

    pygame.quit()


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
