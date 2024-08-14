import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random

pdfFile = PdfPages("output.pdf")

for pltItr in range(10):
    xVals = [x for x in range(40)]
    yVals = [random.randint(50,100) for x in xVals]

    fig, ax = plt.subplots()
    la, = ax.plot(xVals, yVals)
    pdfFile.savefig(fig)

pdfFile.close()
             