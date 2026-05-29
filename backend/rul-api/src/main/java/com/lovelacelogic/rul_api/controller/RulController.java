package com.lovelacelogic.rul_api.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.lovelacelogic.rul_api.service.PythonPredictionService;

@RestController
public class RulController {

    private final PythonPredictionService predictionService;

    public RulController(PythonPredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @GetMapping("/api/rul/health")
    public String healthCheck() {
        return "Aerospace RUL API is running.";
    }

    @GetMapping("/api/rul/predict")
    public String predict(
            @RequestParam(defaultValue = "LSTM") String model
    ) {
        return predictionService.runPrediction(model);
    }
}