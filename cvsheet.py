#python -m streamlit run Cvsheet.py
import cv2
import numpy as np
import ku1
import os
import fitz
import zipfile
import streamlit as st

#################################
webCamFeed = False
#pathImage = "222.png"
cap = cv2.VideoCapture(1)
heightImg = 1100
widthImg = 800
questions = 10
choices = 4
ans = [3,1,1,1,1,1,1,1,1,3]
##################################
def pdftoimg(pdfPath, imagePath):
    pdfDoc = fitz.open(pdfPath)
    for pg in range(pdfDoc.pageCount):
        page = pdfDoc[pg]
        rotate = int(0)
        zoom_x = 1.33333333
        zoom_y = 1.33333333
        mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pix = page.getPixmap(matrix=mat, alpha=False)

        if not os.path.exists(imagePath):
            os.makedirs(imagePath)

        name = pdfPath.split(".")[0]
        pix.writePNG(imagePath + '/' + '%s%s.png' % (name,pg+1))
def zip_file(src_dir):
    zip_name = src_dir +'.zip'
    z = zipfile.ZipFile("zippdf", 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        fpath = dirpath.replace(src_dir,'')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename),fpath+filename)
            print ('==压缩成功==')
    z.close()
def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')
def cvcheck(pathImage):
    a = 1
    count = 0
    while a==1:

        # if webCamFeed:
        #     success, img = cap.read()
        # else:
        img = cv2.imread(pathImage)
        img = cv2.resize(img, (widthImg, heightImg),interpolation = cv2.INTER_AREA)
        #cv2.imshow("Result", img)
        imgFinal = img.copy()
        imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, 10, 70)

        try:
            imgContours = img.copy()
            imgBigContour = img.copy()
            contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
            rectCon = ku1.rectContour(contours)
            biggestPoints = ku1.getCornerPoints(rectCon[0])
            gradePoints = ku1.getCornerPoints(rectCon[1])
            print(gradePoints)

            if biggestPoints.size != 0 and gradePoints.size != 0:
                biggestPoints = ku1.reorder(biggestPoints)
                cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20)
                pts1 = np.float32(biggestPoints)
                pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

                cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)
                gradePoints = ku1.reorder(gradePoints)
                ptsG1 = np.float32(gradePoints)
                #print(ptsG1)
                ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
                matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
                imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

                imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
                imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
                #cv2.imwrite("imgs/" + "6.jpg", imgThresh)
                boxes = ku1.splitBoxes(imgThresh)
                #cv2.imwrite("imgs/" + "7.jpg", boxes)
                #cv2.imshow("Split Test ", boxes[3])
                countR = 0
                countC = 0
                myPixelVal = np.zeros((questions, choices))
                for image in boxes:
                    # cv2.imshow(str(countR)+str(countC),image)
                    totalPixels = cv2.countNonZero(image)
                    myPixelVal[countR][countC] = totalPixels
                    countC += 1
                    if (countC == choices): countC = 0;countR += 1

                myIndex = []
                for x in range(0, questions):
                    arr = myPixelVal[x]
                    myIndexVal = np.where(arr == np.amax(arr))
                    myIndex.append(myIndexVal[0][0])
                #print("USER ANSWERS",myIndex)

                grading = []
                for x in range(0, questions):
                    if ans[x] == myIndex[x]:
                        grading.append(1)
                    else:
                        grading.append(0)
                # print("GRADING",grading)
                score = (sum(grading) / questions) * 100
                # print("SCORE",score)

                ku1.showAnswers(imgWarpColored, myIndex, grading, ans)
                ku1.drawGrid(imgWarpColored)  # DRAW GRID
                imgRawDrawings = np.zeros_like(imgWarpColored)
                ku1.showAnswers(imgRawDrawings, myIndex, grading, ans)
                invMatrix = cv2.getPerspectiveTransform(pts2, pts1)
                imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))

                imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)
                #print(imgRawGrade)
                cv2.putText(imgRawGrade, str(int(score)) + "%", (50, 100)
                            , cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 8)
                #cv2.imwrite("imgs/" +"3.jpg", imgRawGrade)
                invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)
                imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))
                #cv2.imwrite("imgs/" + "4.png", invMatrixG)
                #cv2.imwrite("imgs/" + "5.png", imgInvWarp)

                imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
                imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay,1, 0)

                imageArray = ([img, imgGray, imgCanny, imgContours],
                              [imgBigContour, imgThresh, imgWarpColored, imgFinal])
                #cv2.imshow("Final Result", imgFinal)
        except:
            st.error("Please upload a ZIP file")
            imageArray = ([img, imgGray, imgCanny, imgContours],
                           [imgBlank, imgBlank, imgBlank, imgBlank])

        lables = [["Original", "Gray", "Edges", "Contours"],
                  ["Biggest Contour", "Threshold", "Warpped", "Final"]]

        stackedImage = ku1.stackImages(imageArray, 0.5, lables)
        #cv2.imshow("Result", img)
        #if cv2.waitKey(1) & 0xFF == ord('s'):
        name = pathImage.split("/")[2].split(".")[0]
        cv2.imwrite("results/" + name + ".jpg", imgFinal)
        #cv2.imwrite("imgs/" + str(count+1) + ".jpg", stackedImage)
        #cv2.imshow('Result', stackedImage)
        #cv2.waitKey(300)
        count += 1
        a+=1
def show():
    st.title('Batch marking')
    st.write("""By Gusto""")
    st.info("This software is used to Mark the papers. You can download "
            "dedicated answer sheets. You can scan the completed answer sheet(PDF), upload the ZIP file, and the APP will "
            "automatically grade the papers and output the ZIP(jpg) file"
            )
    col1, col2 = st.columns(2)
    with col1:
        st.header("Answer Sheet Image")
        st.image("answersheet.png",width=200)
    with col2:
        st.header("Filling Example")
        st.image("222.png",width=200)
def downloadzip(imgzip):
    with open(imgzip, "rb") as fp:
        btn = st.download_button(
            label="Download ZIP",
            data=fp,
            file_name="myfile.zip",
            mime="application/zip")
        if btn == True:
            st.success("The zipfile is downloaded successfully")
def downloadimg(imgdl):
    with open(imgdl, "rb") as file:
        btn = st.download_button(
                label="Download answer sheet",
                data=file,
                file_name="Answer Sheet.png",
                mime="image/png"
              )
        if btn == True:
            st.success("The answer sheet is downloaded successfully")
def zipf(path):
    zip_file = path + '.zip'
    z = zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED)
    print(z)
    for path, dirname, file_name in os.walk(path):
        fpath = path.replace(path, '')
        fpath = fpath and fpath + os.sep
        for filename in file_name:
            z.write(os.path.join(path, filename), fpath + filename)
    z.close()
    return zip_file
if __name__ == '__main__':
    show()
    downloadimg("answersheet.png")
    uploaded_file = st.file_uploader("Choose a ZIP file")
    #if uploaded_file is not None:
    st.write(uploaded_file)

    try:
        unzip_file(uploaded_file, "./zippdf")
        path = 'zippdf'
        for file_name in os.listdir(path):
            pdfPath = file_name
            imagePath = './imgs'
            pdftoimg(pdfPath, imagePath)
            #print(file_name)
        #pdfPath = 'opencvsheet.pdf'
        #imagePath = './imgs'
        #pdftoimg(pdfPath, imagePath)
        for file_name in os.listdir("imgs"):
            #print(file_name)
            #cvcheck("./imgs/opencvsheet1.png")
            cvcheck("./imgs/"+file_name)
        imgzip = zipf("results")
        downloadzip(imgzip)

    except:
        st.error("Please upload a ZIP file")

