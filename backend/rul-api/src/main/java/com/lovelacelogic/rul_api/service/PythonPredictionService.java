package com.lovelacelogic.rul_api.service;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import org.springframework.stereotype.Service;

@Service
public class PythonPredictionService {

    public String runPrediction(String modelName) {
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    "/Users/kristinachaleunsak/Desktop/aerospace-rul-prediction/.venv/bin/python",
                    "-m",
                    "src.infer_and_generate_workorders",
                    "--model",
                    modelName
            );

            pb.directory(new java.io.File("/Users/kristinachaleunsak/Desktop/aerospace-rul-prediction"));
            pb.redirectErrorStream(true);

            Process process = pb.start();

            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );

            StringBuilder output = new StringBuilder();
            String line;

            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            process.waitFor();

            return output.toString();

        } catch (Exception e) {
            return "Error running prediction: " + e.getMessage();
        }
    }
}