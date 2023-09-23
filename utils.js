let animationFrameId = null;
let isProcessing = false;
        if (animationFrameId) cancelAnimationFrame(animationFrameId);

        const resultCanvas = document.getElementById('resultCanvas');
        const ctx = resultCanvas.getContext('2d');
        ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);

let stream;
let session

let displayWidth;
let displayHeight;
const classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split('');
const colors = Array.from({ length: classes.length }, () => [Math.random() * 255, Math.random() * 255, Math.random() * 255]);

async function processVideoFrame() {
    const video = document.getElementById('video');
    if (video.paused || video.ended) return;
    const canvas = document.getElementById('resultCanvas');
    const ctx = canvas.getContext('2d');

    if (video.videoWidth === 0 || video.videoHeight === 0) return;
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    const inputTensor = await preprocessVideoFrame(video);
    const modelUrl = 'best416.onnx';
    await main(modelUrl, inputTensor);

    if (isProcessing) requestAnimationFrame(processVideoFrame);
}

async function preprocessVideoFrame(video) {
    let canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    let ctx = canvas.getContext('2d');
    if (video.videoWidth === 0 || video.videoHeight === 0) return;
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let matObj = cv.matFromImageData(imageData);

    let mat = new cv.Mat(matObj.rows, matObj.cols, cv.CV_8UC3);
    cv.cvtColor(matObj, mat, cv.COLOR_RGBA2RGB);
    let blackCanvas = new cv.Mat.zeros(Math.max(mat.cols, mat.rows), Math.max(mat.cols, mat.rows), cv.CV_8UC3);
    let roi = blackCanvas.roi(new cv.Rect(0, 0, mat.cols, mat.rows));
    mat.copyTo(roi);
    const blob = cv.blobFromImage(blackCanvas, 1.0 / 255.0, new cv.Size(416, 416), new cv.Scalar(0, 0, 0), false, false, cv.CV_32F);
    const inputT = new ort.Tensor('float32', blob.data32F, [1, 3, 416, 416]);

    return inputT;
}

async function runObjectDetection(session, inputImage) {
    const [height, width] = [inputImage.height, inputImage.width];
    console.log("Height: ", height)
    console.log("imaage: ", inputImage)

    const length = Math.max(height, width);
    const scale = length / 416;

    const inputTensor = await preprocessVideoFrame(inputImage);
    const { output0 } = await session.run({ images: inputTensor });

    return { output0, scale };
}

function drawBoundingBox(ctx, classId, confidence, x, y, xPlusW, yPlusH) {
    const label = `${classes[classId]} (${confidence.toFixed(2)})`;
    console.log(classId)
    console.log(confidence)

    const color = `rgb(${colors[classId].join(',')})`;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, xPlusW - x, yPlusH - y);
    ctx.font = '16px sans-serif';
    ctx.fillStyle = color;
    ctx.fillText(label, x - 10, y - 10);
}
function transpose(tensor) {
    const [batch, rows, cols] = tensor.dims;
    const transposed = new Float32Array(cols * rows);
    const data = tensor.data;

    for (let row = 0; row < rows; ++row) {
        for (let col = 0; col < cols; ++col) {
            transposed[col * rows + row] = data[row * cols + col];
        }
    }
    return new ort.Tensor(tensor.type, transposed, [cols, rows]);
}

async function main(modelUrl) {
    session = await ort.InferenceSession.create(modelUrl);
    const webcamVideo = document.getElementById('video');
    const canvas = document.getElementById('resultCanvas');
    const ctx = canvas.getContext('2d');

    async function detectFrame() {
        //!!!!
        const xScale = 1;
        const yScale = 1;

        const { output0, scale } = await runObjectDetection(session, webcamVideo);
        console.log(output0)
        console.log("Scale: ", scale)

        const outputData = transpose(output0);

        const rows = outputData.dims[0];

        //test txt file, blob color rgb
        const boxes = [];
        const scores = [];
        const classIds = [];

        for (let i = 0; i < rows; i++) {
            const rowData = outputData.data.slice(i * 30, (i + 1) * 30);
            const classScores = rowData.slice(4);

            const maxScore = Math.max(...classScores);

            const maxClassIndex = classScores.findIndex(score => score === maxScore);

            if (maxScore >= 0.50) {
                const box = [
                    rowData[0] - (0.5 * rowData[2]), rowData[1] - (0.5 * rowData[3]),
                    rowData[2], rowData[3]
                ];
                boxes.push(box);
                scores.push(maxScore);
                classIds.push(maxClassIndex);
            }

            //console.log(scores)
        }

        const canvas = document.getElementById('resultCanvas');
        canvas.width = displayWidth;
        canvas.height = displayHeight;

        ctx.drawImage(webcamVideo, 0, 0, displayWidth, displayHeight);

        if (scores.length > 0) {
            const maxScoreIndex = scores.indexOf(Math.max(...scores));

            console.log("Class ID: ", classIds[maxScoreIndex])

            const box = boxes[maxScoreIndex];
            drawBoundingBox(ctx, classIds[maxScoreIndex], scores[maxScoreIndex],
                Math.round(box[0] * scale * xScale),
                Math.round(box[1] * scale * yScale),
                Math.round((box[0] + box[2]) * scale * xScale),
                Math.round((box[1] + box[3]) * scale * yScale));
        }
        animationFrameId = requestAnimationFrame(detectFrame);
    }

    detectFrame();
}

async function startWebcam() {

    var button = document.querySelector(".start");
    


    if (button.innerHTML === "Start Webcam") {
        button.innerHTML = "Stop cam";
        button.classList.add("pressed");

        const webcamVideo = document.getElementById('video');

        if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
            alert('Your browser does not support webcam access.');
            return;
        }

        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamVideo.srcObject = stream;

        await new Promise((resolve) => {
            webcamVideo.addEventListener('loadedmetadata', resolve);
        });
        webcamVideo.play();

        const targetWidth = 426;
        const targetHeight = 240;

        const widthFactor = targetWidth / webcamVideo.videoWidth;
        const heightFactor = targetHeight / webcamVideo.videoHeight;

        const scale = Math.min(widthFactor, heightFactor);

        displayWidth = webcamVideo.videoWidth * scale;
        displayHeight = webcamVideo.videoHeight * scale;

        video.width = displayWidth;
        video.height = displayHeight;

        console.log(displayWidth)
        console.log(displayHeight)

        const canvas = document.getElementById('resultCanvas');
        canvas.width = displayWidth;
        canvas.height = displayHeight;

        const modelUrl = 'best416.onnx';
        //best12 = 640x640
        //best = 64x64
        isProcessing = true;
        await main(modelUrl);

    } else {
        button.innerHTML = "Start Webcam";
        button.classList.remove("pressed");

        
            let tracks = stream.getTracks();
            tracks.forEach(function(track) {
                track.stop();
            });
            let video = document.getElementById('video');
            video.srcObject = null;
            session = null;
        isProcessing = false;
        if (animationFrameId) cancelAnimationFrame(animationFrameId);

        const resultCanvas = document.getElementById('resultCanvas');
        const ctx = resultCanvas.getContext('2d');
        ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);

        

    }



}
