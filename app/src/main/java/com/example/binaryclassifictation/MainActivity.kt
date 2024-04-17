package com.example.binaryclassifictation

import android.app.Activity
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.binaryclassifictation.databinding.MainActivityBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {


    private lateinit var binding: MainActivityBinding
    private lateinit var  tflite: Interpreter
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = MainActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        tflite= Interpreter(loadModelFile(this)!!)

        binding.button.setOnClickListener {

            val hr = binding.edittext1.text.toString().toInt()
            val day = binding.edittext2.text.toString().toInt()
            // Run inference using a coroutine
            lifecycleScope.launch {
                val prediction = runInference(hr, day)
                updateButton(prediction)
            }

        }
    }

    private fun runInference(hour: Int, day: Int): String {


        val inputArray = Array(1) { intArrayOf(hour, day) }
        val outputArray = Array(1) { FloatArray(1) }

        tflite.run(inputArray, outputArray)

        val pred = outputArray[0][0]
        // Example: Simple usage for classification with a threshold
        return if (pred > 0.5) "Class 1" else "Class 0"

    }

    private suspend fun updateButton(prediction: String) {
        withContext(Dispatchers.Main) {
            binding.textView.text = "Prediction: $prediction"
        }
    }



    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity): MappedByteBuffer? {
        val fileDescriptor = activity.assets.openFd("model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

}