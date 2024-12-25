"use client";

import React, { useState } from "react";
import axios from "axios";

const Page = () => {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setResult("");
    try {
      const response = await axios.post(
        "https://fake-news-detector-ldpo.onrender.com/predict",
        {
          text: inputText,
        }
      );
      setResult(response.data.prediction);
    } catch (err) {
      console.log(err);
      setError(`${err} - Please try again.`);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 text-black">
      <div className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4 max-w-lg w-full">
        <h1 className="text-2xl font-bold text-center mb-6">
          Fake News Detector
        </h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <textarea
            className="w-full p-3 border border-gray-300 rounded-md"
            placeholder="Enter news text here..."
            rows={5}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
          <button
            type="submit"
            className="w-full bg-gray-900 text-white py-2 px-4 rounded hover:bg-black"
          >
            Check News
          </button>
        </form>
        {result && (
          <div className="mt-6 p-4 border rounded bg-green-100 text-green-800">
            Prediction: <strong>{result}</strong>
          </div>
        )}
        {error && (
          <div className="mt-6 p-4 border rounded bg-red-100 text-red-800">
            {error}
          </div>
        )}
      </div>
    </div>
  );
};

export default Page;
