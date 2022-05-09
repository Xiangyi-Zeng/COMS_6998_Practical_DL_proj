let net;
let isModelLoaded = false;
var currentModel = null;
var stopApp = false;

function handleModelChange(){
    let e = document.getElementById('model-selector');
    value = e.options[e.selectedIndex].value;
    if (currentModel != value) {
        currentModel = value;
        stopApp=true;
    }

    isModelLoaded = false;
    //$('#imagePreview').attr("src", "").hide().fadeIn(650);
    //$('.image-section').hide()
    $('#btn-predict').show();
    $('#result').hide()
    $('.loader').hide();

    console.log(currentModel)
}

$(document).ready(async function () {
    
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr("src", e.target.result).hide().fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]); 
        }
    }

    handleModelChange()

    $("#imageUpload").change(function () {

        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('').hide();
        readURL(this);
    });

    // Showing all possible results
    function showResults(results) {
        //console.log(result)
        $('.loader').hide();
        $('#result').empty()
        $('#result').fadeIn(600) //.text(result[0].className + '  -   ' + result[0].probability.toFixed(3));
        for (let idx = 0; idx < results.length; idx++) {
            let item = $("<div id='prediction'></div>")
//             item.html("<p>"+`Prediction: ${results[idx]["className"]} - confidence(${results[idx]["probability"].toFixed(3)})`+"</p>")
            item.html("<p>"+`Prediction: ${results[idx]["className"]} - confidence(${results[idx]["probability"]})`+"</p>")
            $('#result').append(item)
        }
    }

    // preprocess input image
    function preprocess(img)
    {
        let resized = tf.browser.fromPixels(img)
        img = tf.image.resizeBilinear(resized, [224, 224]).div(tf.scalar(255))
        img = tf.cast(img, dtype = 'float32');

        
        /*mean of natural image*/
        let meanRgb = {  red : 0.485,  green: 0.456,  blue: 0.406 }

        /* standard deviation of natural image*/
        let stdRgb = { red: 0.229,  green: 0.224,  blue: 0.225 }

        let indices = [
                    tf.tensor1d([0], "int32"),
                    tf.tensor1d([1], "int32"),
                    tf.tensor1d([2], "int32")
        ];

        /* sperating tensor channelwise and applyin normalization to each chanel seperately*/ 
        let centeredRgb = {
            red: tf.gather(img,indices[0],2)
                    .sub(tf.scalar(meanRgb.red))
                    .div(tf.scalar(stdRgb.red))
                    .reshape([224,224]),
            
            green: tf.gather(img,indices[1],2)
                    .sub(tf.scalar(meanRgb.green))
                    .div(tf.scalar(stdRgb.green))
                    .reshape([224,224]),
            
            blue: tf.gather(img,indices[2],2)
                    .sub(tf.scalar(meanRgb.blue))
                    .div(tf.scalar(stdRgb.blue))
                    .reshape([224,224]),
        }

        /* combining seperate normalized channels*/
        let processedImg = tf.stack([
            centeredRgb.red, centeredRgb.green, centeredRgb.blue
        ]).expandDims();
        //let final_image = tf.expandDims(img, 0)
        return processedImg;
        //return batch_img
    }

    function getTopKClasses(logits, topK) {
        const softmax = tf.softmax(logits);
        const values = softmax.dataSync();
        softmax.dispose();

        const valuesAndIndices = [];
        for (let i = 0; i < values.length; i++) {
            valuesAndIndices.push({value: values[i], index: i});
        }
        valuesAndIndices.sort((a, b) => {
            return b.value - a.value;
        });
        const topkValues = new Float32Array(topK);
        const topkIndices = new Int32Array(topK);
        for (let i = 0; i < topK; i++) {
            topkValues[i] = valuesAndIndices[i].value;
            topkIndices[i] = valuesAndIndices[i].index;
        }

        const topClassesAndProbs = [];
        for (let i = 0; i < topkIndices.length; i++) {
            topClassesAndProbs.push({
            className: IMAGENET_CLASSES[topkIndices[i]],
            probability: topkValues[i]
            });
        }
        return topClassesAndProbs;
    }


    // Predict
    $('#btn-predict').click(async function () {

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Load the model.
        if (isModelLoaded) {
            console.log('Previously loaded');
        } else {
            $('#result').show().fadeIn(600).text('Model is loading, please wait..');
            net = await tf.loadGraphModel(currentModel)
            console.log('Model loaded');
            isModelLoaded = true;
        }

        // Make a prediction through the model on our image.
        const imgEl = document.getElementById('imagePreview');

        let predictions = await net.predict(preprocess(imgEl));
        predictions = predictions.dataSync();
        //console.log(predictions)
        topPredictions = getTopKClasses(predictions, 5);
        console.log(topPredictions)

        showResults(topPredictions);
    });

});
