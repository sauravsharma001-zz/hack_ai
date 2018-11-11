import cv2, keyboard
import numpy as np
from sys import argv

'''
1. This visualizer module has some regions left intentionally modifiable to accommodate
any changes you deem necessary based on your application or personal preference.

2. Additionally, lines that may appear obfuscated due to their pythonic syntax have
verbose versions right above them that elicit the same functionality.

3. For quitting the display, please use the 'q' key on your keyboard. You may also
choose to terminate the program directly through other means.

4. Kindly maintain the directory structure, as this block of code is not robust to
alterations.
'''


def readCSV(filename):

    allImages = list()
    colors = [(0,0,255),(255,0,0),(0,255,0),(255,255,0),(255,0,255),(0,255,255),(128,0,128)]
    lines = None

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()[1:]  # Starts reading from second line
    except Exception as e:
        print("\n\nFile not found")
        print("Please place your file in the same directory or provide an absolute path")
        print("In the event you're using data.csv, please place it in the same directory as this program file")
        exit(0)

    loopIndex=0
    activeState = True

    while activeState:
        # Get current image
        currLine = lines[loopIndex]
        tokens = currLine.strip().split(",")
        fileName = tokens[0]
        image = cv2.imread('input/' + fileName)


        # Get each region
        regions = []
        end = 1
        for i in range(7):
            start = end
            end = start+9
            midList = list(map(int,tokens[start:end]))
            regions.append(midList)
            end+=7

        # Draw polygons
        if image is not None:
            for i, region in enumerate(regions):
                if region[0]==1:
                    # coords = np.array([[region[1],region[2]],[region[3],region[4]],[region[5],region[6]],[region[7],region[8]]])
                    coords = np.array([[region[i],region[i+1]] for i in range(1,9,2)])
                    coords = coords.reshape((-1,1,2))
                    image = cv2.polylines(image, [coords],True,colors[i],3)

            cv2.namedWindow(fileName,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(fileName,(2000,2000))
            cv2.imshow(fileName,image)
            cv2.waitKey(0)

        try:
            if keyboard.is_pressed('q'):
                activeState = False
            if keyboard.is_pressed('d'):
                loopIndex = 0 if loopIndex==len(lines)-1 else (loopIndex+1)
            if keyboard.is_pressed('a'):
                loopIndex = len(lines)-1 if loopIndex==0 else (loopIndex-1)
            cv2.destroyAllWindows()
        except Exception as e:
            print("Error")
            break

    return


def main():
    Guide = ["The program by default attempts to read a file called data.csv",
            "If you chose to call your file something else, please provide it's name followed by .csv as a command line argument",
            "eg:- visualizer.py myfile.csv","\n",
            "The navigation keys have been mapped as follows:",
            "'Q' = terminates the program",
            "'A' = previous image",
            "'D' = next image",
            "Note - pressing any other key will refresh the image"]
    for guidelines in Guide:
        print(guidelines)

    readCSV("extra_docs/Unknown-Training.csv" if len(argv) == 1 else argv[1])


if __name__ == '__main__':
    main()
