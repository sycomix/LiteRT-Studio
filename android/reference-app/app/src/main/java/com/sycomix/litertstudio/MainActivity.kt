package com.sycomix.litertstudio

import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import android.widget.ArrayAdapter
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.sycomix.litertstudio.databinding.ActivityMainBinding
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var modelFile: File? = null

    private val chooseModel = registerForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri: Uri? ->
        if (uri != null) {
            lifecycleScope.launch { importModel(uri) }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.backend.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            listOf("CPU", "GPU"),
        )
        binding.chooseModel.setOnClickListener {
            chooseModel.launch(arrayOf("application/octet-stream", "*/*"))
        }
        binding.run.setOnClickListener { runInference() }
    }

    private suspend fun importModel(uri: Uri) {
        setBusy(true)
        runCatching {
            withContext(Dispatchers.IO) {
                val models = File(filesDir, "models").apply { mkdirs() }
                val destination = File(models, "model.litertlm")
                contentResolver.openInputStream(uri).use { input ->
                    requireNotNull(input) { "Unable to open the selected model" }
                    destination.outputStream().use(input::copyTo)
                }
                destination
            }
        }.onSuccess {
            modelFile = it
            binding.modelStatus.text = "${it.name} · ${it.length() / (1024 * 1024)} MiB"
            binding.run.isEnabled = true
        }.onFailure {
            binding.modelStatus.text = it.message ?: "Model import failed"
        }
        setBusy(false)
    }

    private fun runInference() {
        val model = modelFile ?: return
        val prompt = binding.prompt.text.toString()
        val backendName = binding.backend.selectedItem.toString()
        setBusy(true)
        lifecycleScope.launch {
            val result = runCatching {
                withContext(Dispatchers.IO) {
                    val backend = if (backendName == "GPU") Backend.GPU() else Backend.CPU()
                    val started = SystemClock.elapsedRealtime()
                    Engine(
                        EngineConfig(
                            modelPath = model.absolutePath,
                            backend = backend,
                            cacheDir = cacheDir.absolutePath,
                        ),
                    ).use { engine ->
                        engine.initialize()
                        val loadMs = SystemClock.elapsedRealtime() - started
                        val generateStarted = SystemClock.elapsedRealtime()
                        val response = engine.createConversation().use {
                            it.sendMessage(prompt).toString()
                        }
                        val generationMs = SystemClock.elapsedRealtime() - generateStarted
                        "$response\n\nBackend: $backendName · Load: ${loadMs}ms · Generate: ${generationMs}ms"
                    }
                }
            }
            binding.result.text = result.getOrElse { it.stackTraceToString() }
            setBusy(false)
        }
    }

    private fun setBusy(busy: Boolean) {
        binding.progress.visibility = if (busy) View.VISIBLE else View.GONE
        binding.chooseModel.isEnabled = !busy
        binding.run.isEnabled = !busy && modelFile != null
    }
}
