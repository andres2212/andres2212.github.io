const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let session;
let metadata;

async function fetchMetadata() {
    const response = await fetch('metadata2.yaml');
    const text = await response.text();
    const data = jsyaml.load(text);
    metadata = data.names;
}

async function init() {
    await fetchMetadata();
    await setupModel();
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
        video.addEventListener('loadeddata', () => {
            detectObjects();
        });
    });
}

async function setupModel() {
    const options = { executionProviders: ['cpu'] };
    session = await ort.InferenceSession.create('best12.onnx', options);
}

//i changed from 640 to 64 and also changed the model just in case
async function detectObjects() {
    ctx.drawImage(video, 0, 0, 640, 640);
    const imageData = ctx.getImageData(0, 0, 640, 640);

    // Convert the input data to Float32Array and normalize it to the range [0, 1]
    const numPixels = imageData.data.length / 4;
    const imageDataArray = new Float32Array(numPixels * 3);
    for (let i = 0; i < numPixels; i++) {
        imageDataArray[i] = imageData.data[i * 4] / 255.0;
        imageDataArray[numPixels + i] = imageData.data[i * 4 + 1] / 255.0;
        imageDataArray[numPixels * 2 + i] = imageData.data[i * 4 + 2] / 255.0;
    }

    // Create the input tensor using the converted and normalized data
    const inputTensor = new ort.Tensor('float32', imageDataArray, [1, 3, 640, 640]);
    //let inputTensor
    //inputTensor = new ort.Tensor('float32', new Float32Array(1 * 3 * 64 * 64), [1, 3, 64, 64]);

    // Run the model with the input tensor
    const inputName = session.inputNames[0];
    const outputMap = await session.run({ [inputName]: inputTensor });
    console.log("Outputmap: ",outputMap)
    console.log("Outputmap length: ",outputMap.length)
    
    const outputNames = session.outputNames;
    //console.log(outputNames);
    const outputData = outputMap[outputNames[0]].data;
    console.log(outputData)
    console.log(outputData.length)
    

    //const outputData = outputMap.values().next().value.data;

    processBoundingBoxes(outputData);

    requestAnimationFrame(detectObjects);
}





function drawBoundingBox(ctx, classId, confidence, x, y, xPlusW, yPlusH) {
    const label = `${metadata[classId]} (${confidence.toFixed(2)})`;
    const color = `rgb(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255})`;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, xPlusW - x, yPlusH - y);
    ctx.font = '14px Arial';
    ctx.fillStyle = color;
    ctx.fillText(label, x - 10, y - 10);
    //console.log(label)
    //console.log(classId)
    
}

function processBoundingBoxes(outputData) {
    const boxes = [];
    const scores = [];
    const classIds = [];



    for (let i = 0; i < outputData.length / 7; i++) {
        const row = outputData.slice(i * 7, i * 7 + 7);
        const [x, y, w, h, _, ...classScores] = row;
        const maxScore = Math.max(...classScores);
        const maxClassIndex = classScores.indexOf(maxScore);

        if (maxScore >= 0.25) {
            const box = [x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h];
            boxes.push(box);
            scores.push(maxScore);
            classIds.push(maxClassIndex);
        }
    }

    const indices = nms(boxes, scores, 0.25, 0.45);

    //console.log('classIds', classIds);
    //console.log('classIds', classIds[4000]);
    
    //console.log('scores', scores);
    //console.log('boxes', boxes);
    //console.log('indices', indices);


    if (scores.length > 0) {
        const maxConfidenceIndex = scores.indexOf(Math.max(...scores));
        if (indices.includes(maxConfidenceIndex)) {
            const [x, y, xPlusW, yPlusH] = boxes[maxConfidenceIndex].map(v => v * 640);
            drawBoundingBox(ctx, classIds[maxConfidenceIndex], scores[maxConfidenceIndex], x, y, xPlusW, yPlusH);
        }
    }
}

function nms(boxes, scores, scoreThreshold, nmsThreshold) {
    const indices = scores.reduce((indices, score, i) => {
        if (score >= scoreThreshold) {
            indices.push(i);
        }
        return indices;
    }, []);

    indices.sort((a, b) => scores[b] - scores[a]);

    const picked = [];

    while (indices.length > 0) {
        const i = indices.shift();
        picked.push(i);

        const boxA = boxes[i];

        indices.forEach((j, k) => {
            const boxB = boxes[j];
            const iou = computeIOU(boxA, boxB);

            if (iou >= nmsThreshold) {
                indices.splice(k, 1);
            }
        });
    }

    return picked;
}

function computeIOU(boxA, boxB) {
    const [x1A, y1A, x2A, y2A] = boxA;
    const [x1B, y1B, x2B, y2B] = boxB;

    const x1 = Math.max(x1A, x1B);
    const y1 = Math.max(y1A, y1B);
    const x2 = Math.min(x2A, x2B);
    const y2 = Math.min(y2A, y2B);

    const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const boxAArea = (x2A - x1A) * (y2A - y1A);
    const boxBArea = (x2B - x1B) * (y2B - y1B);

    const unionArea = boxAArea + boxBArea - intersectionArea;

    return intersectionArea / unionArea;
}


init();

