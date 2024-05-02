package com.example.binaryclassifictation

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.binaryclassifictation.databinding.MainActivityBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


class MainActivity : AppCompatActivity() {
    // Define your model-specific parameters
    val NUM_EPOCHS = 10
    val BATCH_SIZE = 2 // Adjust as needed
    val NUM_TRAINING_SAMPLES = 10 // Total number of training samples
    val NUM_BATCHES = NUM_TRAINING_SAMPLES / BATCH_SIZE
    val NUM_FEATURES = 2
    // Initialize lists for training data
    val trainFeatureBatches = mutableListOf<FloatBuffer>()
    val trainLabelBatches = mutableListOf<FloatBuffer>()
    private lateinit var binding: MainActivityBinding
    private var tflite: Interpreter? = null


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = MainActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        tflite = createInterpreter()

        binding.button.setOnClickListener {

            val hr = binding.edittext1.text.toString().toInt()
            val day = binding.edittext2.text.toString().toInt()
            // Run inference using a coroutine
            lifecycleScope.launch {
                val prediction = runInference(hr, day)
                updateButton(prediction)
            }

        }

        binding.trainbutton.setOnClickListener {
            // Run inference using a coroutine
            lifecycleScope.launch {
                train()
            }

        }

        binding.predictbutton.setOnClickListener{
            // Run inference using a coroutine
            val hr = binding.edittext3.text.toString().toInt()
            val day = binding.edittext4.text.toString().toInt()
            lifecycleScope.launch {
                predict(hr, day)
            }
        }
        buildTrainData()
    }

    private fun runInference(hour: Int, day: Int): String {


        val inputArray = Array(1) { intArrayOf(hour, day) }
        val outputArray = Array(1) { FloatArray(1) }

        tflite?.run(inputArray, outputArray)

        val pred = outputArray[0][0]
        // Example: Simple usage for classification with a threshold
        return if (pred > 0.5) "Class 1" else "Class 0"

    }

    private suspend fun updateButton(prediction: String) {
        withContext(Dispatchers.Main) {
            binding.textView.text = "Prediction: $prediction"
        }
    }

    private suspend fun train() {
        withContext(Dispatchers.Default) {
            trainData(trainFeatureBatches, trainLabelBatches)
            //binding.textView.text = "Prediction: $prediction"
        }
    }


    private suspend fun predict(hr: Int, day: Int) {
        val trainFeatures = ByteBuffer.allocateDirect(NUM_FEATURES * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

        val hrValue = hr.toFloat()
        val dayValue = day.toFloat()

        trainFeatures.put(hrValue)
        trainFeatures.put(dayValue)
        val inputs: MutableMap<String, Any> = HashMap()
        inputs["inputs"] = trainFeatures
        val outputs: MutableMap<String, Any> = HashMap()
        val pred = FloatBuffer.allocate(1)
        outputs["logits"] = pred

        tflite?.runSignature(inputs, outputs, "infer")

        withContext(Dispatchers.Default) {
            binding.resultView.text = pred[0].toString()
            println(pred)
        }

    }

    private fun trainData(td : List<FloatBuffer>, tb: List<FloatBuffer>){
        // Run training for a few steps.
        // Run training for a few steps.
        val losses = FloatArray(NUM_EPOCHS)
        for (epoch in 0 until NUM_EPOCHS) {
            for (batchIdx in 0 until NUM_BATCHES) {
                val inputs: MutableMap<String, Any> = HashMap()
                inputs["x"] = td[batchIdx]
                inputs["y"] = tb[batchIdx]
                val outputs: MutableMap<String, Any> = HashMap()
                val loss = FloatBuffer.allocate(1)
                outputs["loss"] = loss

                tflite?.runSignature(inputs, outputs, "train")

                // Record the last loss.
                if (batchIdx == NUM_BATCHES - 1) losses[epoch] = loss[0]
            }

            // Print the loss output for every 10 epochs.
            if ((epoch + 1) % 10 == 0) {
                println(
                    "Finished " + (epoch + 1) + " epochs, current loss: " + losses[epoch]
                )
            }
        }

    }

    fun assignStatus(hr: Int, day: Int, centerHr: Int, centerDay: Int, radius: Int): Int {
        val distanceFromCenter = kotlin.math.sqrt(((hr - centerHr) * (hr - centerHr) + (day - centerDay) * (day - centerDay)).toDouble())
        return if (distanceFromCenter <= radius) 1 else 0
    }


    private fun buildTrainData(){

       // Assuming hrValues and dayValues are your training data (features)
        for (i in 0 until NUM_BATCHES) {
            val trainFeatures = ByteBuffer.allocateDirect(NUM_FEATURES * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            val trainLabels = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder()).asFloatBuffer()

            // Fill the data values (replace with your actual data)
            for (j in 1 until NUM_FEATURES) {
                val hrValue = 10f // Your hour feature value
                val dayValue = 4f // Your day feature value

                trainFeatures.put(hrValue)
                trainFeatures.put(dayValue)

                val label = 1f // Your binary label (0 or 1)
                trainLabels.put(label)
            }

            trainFeatureBatches.add(trainFeatures.rewind() as FloatBuffer)
            trainLabelBatches.add(trainLabels.rewind() as FloatBuffer)

        }

        printTrainingData(trainFeatureBatches)
        printTrainingData(trainLabelBatches)
    }


    private fun printTrainingData(list : List<FloatBuffer>){
        for (batch in list) {
            val featureValues = mutableListOf<Float>()
            while (batch.hasRemaining()) {
                featureValues.add(batch.get())
            }
            println(featureValues.joinToString(", "))
        }
    }


    private fun createInterpreter(): Interpreter? {
        try {
            // Load the model
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options()
           val interpreter = Interpreter(modelBuffer, options)

            // Print model info
            val inputSize: Int = interpreter.getInputTensor(0).shape()[1]
            val outputSize: Int = interpreter.getOutputTensor(0).shape()[1]
            Log.i(
                "TAG",
                "Loaded TensorFlow Lite model with input size $inputSize and output size $outputSize"
            )
            return interpreter
        } catch (e: IOException) {
            Log.e("TAG", "Failed to load TensorFlow Lite model", e)
        }
        return null
    }

    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd("modeldevlearning.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /*
private suspend fun onDeviceLearning() {
    val hmin=8
    val hmax=24
    val daymin=1
    val daymax=7


    val sample_size = 400
       val NUM_EPOCHS = 100
        val BATCH_SIZE = 8
        val NUM_FEATURES = 2  // Hour and Day
        val NUM_TRAININGS = 400 // Adjust if needed
        val NUM_BATCHES = NUM_TRAININGS / BATCH_SIZE //50
    val traindataBatches: MutableList<FloatBuffer> = ArrayList(NUM_BATCHES)
    val trainLabelBatches: MutableList<FloatBuffer> = ArrayList(NUM_BATCHES)




    val hr = IntArray(sample_size) { Random.nextInt(hmin, hmax) }
    val day = IntArray(sample_size) { Random.nextInt(daymin, daymax + 1) }

    //loop for 50 times
    for (i in 0 until NUM_BATCHES) {
        val trainData = FloatBuffer.allocate(BATCH_SIZE * NUM_FEATURES) //each times alloc 8*2
        val trainLabels = FloatBuffer.allocate(BATCH_SIZE) // one label alloc 8


        val hour = IntArray(sample_size) { Random.nextInt(hmin, hmax) }
        val day = IntArray(sample_size) { Random.nextInt(daymin, daymax + 1) }

        //val status = assignStatus(hour, day, 20, 5, 3) // Call the assignStatus function

        //traindataBatches.add(hour)

        //trainLabels.put(status.toFloat())
    }


    /*
        val trainDataBatches = ArrayList<FloatBuffer>(NUM_BATCHES)
        val trainLabelBatches = ArrayList<FloatBuffer>(NUM_BATCHES)

        // Prepare training batches.
        for (i in 0 until NUM_BATCHES) {
            val trainData = FloatBuffer.allocate(BATCH_SIZE * NUM_FEATURES)
            val trainLabels =
                FloatBuffer.allocate(BATCH_SIZE * 10) // Replace 10 with the number of output classes

            // Fill the data values... (You'll need to replace this)
            trainDataBatches.add(trainData)
            trainLabelBatches.add(trainLabels)
        }

        // Run training for a few steps.
        val losses = FloatArray(NUM_EPOCHS)
        for (epoch in 0 until NUM_EPOCHS) {
            for (batchIdx in 0 until NUM_BATCHES) {
                val inputs = HashMap<String, Any>()
                inputs["x"] = trainDataBatches[batchIdx]
                inputs["y"] = trainLabelBatches[batchIdx]

                val outputs = HashMap<String, Any>()
                val loss = FloatBuffer.allocate(1)
                outputs["loss"] = loss

                tflite.runSignature(inputs, outputs, "train")

                // Record the last loss.
                if (batchIdx == NUM_BATCHES - 1) losses[epoch] = loss.get(0)
            }

            // Print the loss output for every 10 epochs.
            if ((epoch + 1) % 10 == 0) {
                println("Finished ${epoch + 1} epochs, current loss: ${loss.get(0)}")
            }
        }
*/
        // ...
    }
*/


}