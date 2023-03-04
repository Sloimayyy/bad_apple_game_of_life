import math

import numpy as np
from PIL import Image
from moviepy.video.VideoClip import VideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from numba import cuda





def main():
    badAppleGameOfLife()





def badAppleGameOfLife():

    ### Setup
    badAppleClip: VideoFileClip = VideoFileClip("bad_apple.mp4")
    badAppleFrameGen = badAppleClip.iter_frames()
    ## Back and forth grids
    grid1: np.ndarray = np.zeros(shape=badAppleClip.size[::-1], dtype=np.bool)
    gridSize: tuple[int, int] = grid1.shape
    grid1 = np.random.randint(2, size=gridSize, dtype=np.bool)
    grid2: np.ndarray = np.zeros(shape=gridSize, dtype=np.bool)
    ## Options: (iterationCount, gridSizeX, gridSizeY, iterationIndex)
    options: np.array = np.array([1000, *gridSize, 0])


    ### GOL
    @cuda.jit
    def gol(grid1: np.ndarray, grid2: np.ndarray, options: np.ndarray):
        cellX, cellY = cuda.grid(2)

        gridSizeX: int = options[1]
        gridSizeY: int = options[2]

        ## Stop the trhead if its outside of the grid
        if not (cellX < gridSizeX or cellY < gridSizeY):
            return

        ## Swap them every iteration, like a frame buffer
        iterIndex: int = options[3]
        sourceGrid: np.ndarray = grid1 if iterIndex % 2 == 0 else grid2
        destGrid: np.ndarray = grid2 if iterIndex % 2 == 0 else grid1

        ## Calculate neighbor count
        neighborCount: int = 0
        for dx in range(-1, 1+1):
            for dy in range(-1, 1+1):
                ## Get the current cell checked XY and see if its within the grid
                if dx == 0 and dy == 0:
                    continue
                cellCheckedX: int = cellX + dx
                cellCheckedY: int = cellY + dy
                if not (0 <= cellCheckedX < gridSizeX):
                    continue
                if not (0 <= cellCheckedY < gridSizeY):
                    continue
                ##
                neighborCount += sourceGrid[cellCheckedX, cellCheckedY]

        ## Apply the rules of GOL and write in the dest grid
        currentCellValue: int = sourceGrid[cellX, cellY]

        destGrid[cellX, cellY] = 0
        if currentCellValue == 1:
            if neighborCount == 2 or neighborCount == 3:
                destGrid[cellX, cellY] = 1
        else:
            if neighborCount == 3:
                destGrid[cellX, cellY] = 1


    threadsPerBlock: tuple[int, int] = (8, 8)
    blocksPerGrid: tuple[int, int] = (
        math.ceil(gridSize[0] / threadsPerBlock[0]),
        math.ceil(gridSize[1] / threadsPerBlock[1]),
    )
    iterIndex: int = -1
    def renderFrameBadApple(t: float) -> np.ndarray:
        """
        Renders Bad Apple GOL
        """
        nonlocal iterIndex
        iterIndex += 1
        options[3] = iterIndex
        gol[blocksPerGrid, threadsPerBlock](grid1, grid2, options)
        ## Add the bad apple frame to the grid
        gridToVisualize: np.array = grid2 if iterIndex % 2 == 0 else grid1
        badAppleFrame = None
        try:
            badAppleFrame = next(badAppleFrameGen)
        except:
            pass
        if badAppleFrame is not None:
            frameToAdd = np.array(Image.fromarray(badAppleFrame).convert('1'))
            gridToVisualize[frameToAdd] = 1

        img = Image.fromarray(gridToVisualize).convert('RGB')
        out = np.array(img)
        return out

    def renderFrameGlider(t: float):
        """
        Renders the glider scene in the intro of the video
        """
        nonlocal iterIndex
        iterIndex += 1
        options[3] = iterIndex
        gol[blocksPerGrid, threadsPerBlock](grid1, grid2, options)
        gridToVisualize: np.array = grid2 if iterIndex % 2 == 0 else grid1
        img = Image.fromarray(gridToVisualize).convert('RGB')
        out = np.array(img)
        return out


    golClip = VideoClip(renderFrameBadApple, duration=badAppleClip.duration + 20)
    #golClip = VideoClip(renderFrameGlider, duration=60)

    ## Render epic bideo
    golClip.write_videofile('vid.avi', fps=badAppleClip.fps, bitrate='100000k', codec='mpeg4')




if __name__ == "__main__":
    main()
