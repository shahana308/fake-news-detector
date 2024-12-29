"use client";

import React, { useState } from "react";
import axios from "axios";

const Page = () => {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false); // State for loader

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setResult("");
    setLoading(true); // Set loading to true
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
    } finally {
      setLoading(false); // Set loading to false after request completes
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
            className="w-full bg-gray-900 text-white py-2 px-4 rounded hover:bg-black flex items-center justify-center"
            disabled={loading} // Disable button while loading
          >
            {loading ? (
              <svg
                className="animate-spin h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
            ) : (
              "Check News"
            )}
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
