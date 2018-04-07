// 要被訓練的參數，這個參數隨著訓練次數增加會越來越準確
const trainingAnswer = tf.variable(tf.scalar(Math.random()));

function generateData(numPoints, answer) {
    return tf.tidy(() => {
        // 產生常態分佈的隨機資料
        const xs = tf.randomNormal([numPoints], -1, 1);
        // 套用正確模型產生答案
        const ans = tf.scalar(answer);
        const ys = ans.mul(xs);
        // 回傳訓練資料與答案
        return {
            xs,
            ys
        };
    })
}

function predict(x) {
    return tf.tidy(() => {
        return trainingAnswer.mul(x)
    });
}

function loss(predictions, labels) {
    const meanSquareError = predictions.sub(labels).square().mean();
    return meanSquareError;
}

function train(xs, ys, numIterations = 75) {

    const learningRate = 0.5;
    const optimizer = tf.train.sgd(learningRate);

    for (let iter = 0; iter < numIterations; iter++) {
        optimizer.minimize(() => {
            const predsYs = predict(xs);
            return loss(predsYs, ys);
        });
    }
}

async function learnCoefficients(dataCount, iterations) {
    const correctAnswer = 2; // 正確答案
    const trainingData = generateData(dataCount, 2);

    console.log('Before Training: ', await trainingAnswer.data());

    // Train the model!
    await train(trainingData.xs, trainingData.ys, iterations);

    // 印出訓練結果
    console.log('After Training: ', await trainingAnswer.data());
}

learnCoefficients(100, 1000);