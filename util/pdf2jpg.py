import os
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

students = 5

file_path = "C:/grading_test/pdfFolder/class/"
file_name = "a_test.pdf"
outputFolder_path = file_path + "jpg/" + file_name[:-4] + "/"


def createFolder(directory_path):
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    except OSError:
        print('Error: Cannot Creating directory. ' + directory_path)

createFolder(outputFolder_path)

pages = convert_from_bytes(open(file_path + file_name, 'rb').read(), dpi=600)

for i, page in enumerate(pages):
    page.save(outputFolder_path + file_name[:-4] + "_" + str(i) + ".jpg", "JPEG")
