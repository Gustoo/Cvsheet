import cv2
import numpy as np
import ku1
import os
import fitz
import zipfile
import streamlit as st
import shutil

#################################
webCamFeed = False
heightImg = 1100
widthImg = 800
questions = 10
choices = 4
ans = [3,1,1,1,1,1,1,1,1,3]
anss = []
##################################

def clean_folder(folder):
    """清空文件夹，避免旧文件干扰"""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

def pdftoimg(pdfPath, imagePath):
    pdfDoc = fitz.open(pdfPath)
    base_name = os.path.splitext(os.path.basename(pdfPath))[0]
    zoom_x = 1.33333333
    zoom_y = 1.33333333
    mat = fitz.Matrix(zoom_x, zoom_y)

    for pg in range(len(pdfDoc)):
        page = pdfDoc[pg]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        os.makedirs(imagePath, exist_ok=True)
        out_path = os.path.join(imagePath, f'{base_name}_pg{pg+1}.png')
        pix.save(out_path)

def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        fz.extractall(dst_dir)
        fz.close()
    else:
        st.error("不是合法的ZIP文件")

def cvcheck(pathImage):
    try:
        img = cv2.imread(pathImage)
        if img is None:
            st.warning(f"无法读取图片：{pathImage}")
            return

        img = cv2.resize(img, (widthImg, heightImg), interpolation=cv2.INTER_AREA)
        imgFinal = img.copy()
        imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, 10, 70)

        imgContours = img.copy()
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
        rectCon = ku1.rectContour(contours)

        if len(rectCon) < 2:
            st.warning(f"图片 {os.path.basename(pathImage)} 未检测到答题卡，已跳过")
            return

        biggestPoints = ku1.getCornerPoints(rectCon[0])
        gradePoints = ku1.getCornerPoints(rectCon[1])

        if biggestPoints.size != 0 and gradePoints.size != 0:
            biggestPoints = ku1.reorder(biggestPoints)
            pts1 = np.float32(biggestPoints)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            gradePoints = ku1.reorder(gradePoints)
            ptsG1 = np.float32(gradePoints)
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

            boxes = ku1.splitBoxes(imgThresh)
            countR = 0
            countC = 0
            myPixelVal = np.zeros((questions, choices))

            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if countC == choices:
                    countC = 0
                    countR += 1

            myIndex = []
            for x in range(questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])

            grading = []
            for x in range(questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)

            score = (sum(grading) / questions) * 100

            ku1.showAnswers(imgWarpColored, myIndex, grading, ans)
            imgRawDrawings = np.zeros_like(imgWarpColored)
            ku1.showAnswers(imgRawDrawings, myIndex, grading, ans)
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1)
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))

            imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)
            cv2.putText(imgRawGrade, f"{int(score)}%", (50, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 8)
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

            name = os.path.splitext(os.path.basename(pathImage))[0]
            cv2.imwrite(f"results/{name}.jpg", imgFinal)
            st.success(f"✅ {name} 批阅完成 | 得分：{int(score)}%")

    except Exception as e:
        st.error(f"处理失败：{str(e)}")

def show():
    global ans, anss
    st.title('📝 答题卡自动批阅系统')
    st.write("By Leo&Gusto")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("标准答题卡")
        st.image("answersheet.png", width=250)
    with col2:
        st.subheader("填写示例")
        st.image("222.png", width=250)

    st.info("输入标准答案，格式示例：3,1,1,1,1,1,1,1,1,3")
    st.info("A=0  B=1  C=2  D=3")

    Answers = st.text_input('请输入标准答案', '3,1,1,1,1,1,1,1,1,3')
    anss = []
    try:
        for i in Answers.split(","):
            anss.append(int(i.strip()))
        if len(anss) == 10:
            ans = anss
            st.success("✅ 答案设置成功！")
        else:
            st.error("必须输入10个答案！")
    except:
        st.error("格式错误！请用英文逗号分隔数字")

def downloadimg(imgdl):
    if os.path.exists(imgdl):
        with open(imgdl, "rb") as file:
            st.download_button(
                label="📥 下载空白答题卡",
                data=file,
                file_name="Answer_Sheet.png",
                mime="image/png"
            )

def zipf(path):
    zip_file = "批阅结果.zip"
    if os.path.exists(zip_file):
        os.remove(zip_file)

    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, path)
                z.write(file_path, arcname)
    return zip_file

if __name__ == '__main__':
    show()
    downloadimg("answersheet.png")

    # 每次上传前清空临时目录，避免冲突
    for folder in ["results", "imgs", "zippdf"]:
        clean_folder(folder)

    uploaded_file = st.file_uploader("📁 上传包含PDF的ZIP压缩包")

    if uploaded_file is not None:
        if st.button("🚀 开始自动批阅", type="primary"):
            temp_zip = "temp_upload.zip"
            with open(temp_zip, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("正在处理..."):
                unzip_file(temp_zip, "zippdf")

                # PDF转图片
                pdfs = [f for f in os.listdir("zippdf") if f.lower().endswith(".pdf")]
                for pdf in pdfs:
                    pdftoimg(f"zippdf/{pdf}", "imgs")

                # 批阅所有图片
                imgs = [f for f in os.listdir("imgs") if f.endswith((".png", ".jpg"))]
                for img in imgs:
                    cvcheck(f"imgs/{img}")

                # 打包下载
                if len(os.listdir("results")) > 0:
                    zip_path = zipf("results")
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label="📥 下载全部批阅结果",
                            data=f,
                            file_name="批阅结果.zip",
                            mime="application/zip"
                        )
                else:
                    st.warning("未生成任何结果")
