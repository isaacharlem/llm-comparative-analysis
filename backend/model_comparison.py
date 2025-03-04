# model_comparison.py
# 
# This file is licensed under the BSD 3-Clause License.
# Copyright (c) 2024, UChicago Data Science Institute
# See the LICENSE-BSD file for details.

import os
import json
import time
import asyncio
import base64
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm
import requests  # for embeddings API calls (synchronous)
import httpx     # for asynchronous generation API calls (async)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the reports directory exists
REPORTS_FOLDER = "/reports"
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Base URL for the Ollama API (the Ollama service runs at port 11434 in docker-compose)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
# Embedding model name to use for generating embeddings (pulled in Ollama)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "smollm:135m")
# Summarization model is fixed to phi3:mini
SUMMARY_MODEL = "phi3:mini"

def current_timestamp():
    """Return the current time as a formatted string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def call_ollama_embeddings(text: str) -> np.ndarray:
    """
    Call the Ollama embedding API to get the embedding vector for the given text.
    Returns a numpy array representing the embedding.
    """
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
            "model": EMBEDDING_MODEL,
            "input": text
        })
        resp.raise_for_status()
        data = resp.json()
        vector = data["embeddings"][0]
        return np.array(vector, dtype=float)
    except Exception as e:
        logger.error(f"Embedding API call failed: {e}")
        return np.array([])

def generate_embeddings(responses: list) -> list:
    """Generate embeddings for all responses using the Ollama embedding model."""
    embeddings = []
    for resp in responses:
        emb = call_ollama_embeddings(resp)
        embeddings.append(emb)
    return embeddings

def calculate_similarity_matrix(embeddings: list) -> np.ndarray:
    """Calculate pairwise cosine similarity matrix for the given embeddings."""
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = compute_cosine_similarity(embeddings[i], embeddings[j])
    return sim_matrix

def rank_responses_by_similarity(sim_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Rank responses by total similarity to others (excludes self-similarity).
    Returns indices sorted by similarity and the array of total similarities.
    """
    n = sim_matrix.shape[0]
    total_similarity = np.zeros(n)
    for i in range(n):
        row = np.delete(sim_matrix[i], i)
        total_similarity[i] = np.sum(np.abs(row))
    ranked_indices = np.argsort(total_similarity)
    return ranked_indices, total_similarity

def best_response(responses: list, total_similarity: np.ndarray) -> str:
    """Return the response with the highest total similarity (most consistent)."""
    if total_similarity.size == 0:
        return ""
    best_index = int(np.argmax(total_similarity))
    return responses[best_index]

def save_embeddings_plot(embeddings: list, output_path: str) -> None:
    """Save a 2D PCA plot of the response embeddings."""
    if not embeddings:
        return
    pca = PCA(n_components=2)
    emb_array = np.array(embeddings)
    reduced = pca.fit_transform(emb_array)
    plt.figure(figsize=(6, 5))
    plt.scatter(reduced[:, 0], reduced[:, 1], c='blue', alpha=0.7)
    for idx, (x, y) in enumerate(reduced):
        plt.annotate(str(idx), (x, y), textcoords="offset points", xytext=(5, 5), ha='center')
    plt.title("PCA of Response Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_similarity_distribution(sim_matrix: np.ndarray, output_path: str) -> None:
    """Save a histogram plot of similarity values (off-diagonal of similarity matrix)."""
    if sim_matrix.size == 0:
        return
    n = sim_matrix.shape[0]
    sim_vals = [sim_matrix[i, j] for i in range(n) for j in range(n) if i != j]
    plt.figure(figsize=(6, 4))
    sns.histplot(sim_vals, bins=10, color='green', kde=False)
    plt.title("Distribution of Pairwise Similarities")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_similarity_heatmap(sim_matrix: np.ndarray, output_path: str) -> None:
    """Save a heatmap visualization of the similarity matrix."""
    if sim_matrix.size == 0:
        return
    plt.figure(figsize=(6, 5))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Similarity Matrix Heatmap")
    plt.xlabel("Response Index")
    plt.ylabel("Response Index")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_total_similarity_bar_chart(total_similarity: np.ndarray, output_path: str) -> None:
    """Save a bar chart of total similarity for each response."""
    if total_similarity.size == 0:
        return
    plt.figure(figsize=(6, 4))
    indices = np.arange(len(total_similarity))
    plt.bar(indices, total_similarity, color='skyblue')
    plt.xlabel("Response Index")
    plt.ylabel("Total Similarity")
    plt.title("Total Similarity per Response")
    plt.xticks(indices)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def image_to_base64(image_path: str) -> str:
    """Convert an image file to a Base64 data URI string."""
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def assess_consistency(sim_matrix: np.ndarray, responses: list) -> dict:
    """
    Compute consistency stats for each response:
    total_similarity, min_similarity, max_similarity, variance.
    """
    stats = {}
    n = len(responses)
    for i in range(n):
        sims = np.delete(sim_matrix[i], i)
        stats[i] = {
            'response': responses[i],
            'total_similarity': float(np.sum(np.abs(sims))),
            'min_similarity': float(np.min(sims)) if sims.size > 0 else 0.0,
            'max_similarity': float(np.max(sims)) if sims.size > 0 else 0.0,
            'variance': float(np.var(sims)) if sims.size > 0 else 0.0
        }
    return stats

def compute_bleu(reference: str, candidate: str) -> float:
    """Compute BLEU score for a single candidate against a reference."""
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        smoothie = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
        return score
    except Exception as e:
        logger.error(f"BLEU computation error: {e}")
        return 0.0

def compute_rouge_l(reference: str, candidate: str) -> float:
    """Compute ROUGE-L F1 score for a single candidate vs reference."""
    ref_words = reference.split()
    cand_words = candidate.split()
    m, n = len(ref_words), len(cand_words)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if ref_words[i] == cand_words[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs_length = dp[m][n]
    if lcs_length == 0:
        return 0.0
    recall = lcs_length / m if m > 0 else 0.0
    precision = lcs_length / n if n > 0 else 0.0
    if recall + precision == 0:
        return 0.0
    f1 = (2 * recall * precision) / (recall + precision)
    return f1

async def generate_response_from_ollama(model: str, prompt: str, ws=None, i=0, n=1) -> str:
    """
    Call the Ollama API to generate a response for the given prompt and model.
    Uses non-stream mode (stream: false) to receive a complete response.
    Returns the generated text (or an empty string on failure).
    If a WebSocket (ws) is provided, sends progress updates with timestamps.
    """
    max_retries = 5
    delay = 3
    timeout = httpx.Timeout(500.0)
    for attempt in range(max_retries):
        try:
            if ws:
                await ws.send_json({
                    "event": "progress",
                    "message": f"[{current_timestamp()}] Model {model}: Attempt {i+1}/{n} starting generation..."
                })
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{OLLAMA_URL}/api/generate", json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                })
                resp.raise_for_status()
                data = resp.json()
                response_text = data.get("response", "")
                logger.info(f"Generated response from model {model}: {response_text[:50]}...")
                if ws:
                    await ws.send_json({
                        "event": "progress",
                        "message": f"[{current_timestamp()}] Model {model}: Generation complete."
                    })
                return response_text
        except httpx.ConnectError as ce:
            logger.warning(f"Attempt {attempt+1}/{max_retries} - ConnectError: {ce}")
            if ws:
                await ws.send_json({
                    "event": "progress",
                    "message": f"[{current_timestamp()}] Model {model}: Attempt {attempt+1} failed with ConnectError. Retrying..."
                })
            await asyncio.sleep(delay)
        except Exception as e:
            logger.exception(f"Generation API call failed for model {model}: {e}")
            if ws:
                await ws.send_json({
                    "event": "progress",
                    "message": f"[{current_timestamp()}] Model {model}: Attempt {attempt+1} failed with error: {str(e)}."
                })
            return ""
    logger.error(f"All retries failed for model {model}")
    if ws:
        await ws.send_json({
            "event": "progress",
            "message": f"[{current_timestamp()}] Model {model}: All retries failed."
        })
    return ""

async def generate_summary_comparison(summary_metrics: list, ws=None) -> str:
    """
    Generate a concise summary comparing the models based on summary_metrics.
    This function builds a prompt from the summary_metrics list and calls the summarization LLM.
    The summarization model is fixed to 'phi3:mini'.
    """
    prompt_lines = ["Summarize the following comparative metrics for multiple LLM models:"]
    for metrics in summary_metrics:
        line = (f"Model: {metrics['model']}. "
                f"Avg Total Similarity Between Responses: {metrics['avg_total_similarity']:.3f}. "
                f"Avg Variance Between Responses: {metrics['avg_variance']:.3f}. "
                f"Best Total Similarity: {metrics['best_total_similarity']:.3f}.")
        if metrics.get("avg_bleu") is not None:
            line += f" Avg BLEU between responses and best response: {metrics['avg_bleu']:.3f}."
        if metrics.get("avg_rougeL") is not None:
            line += f" Avg ROUGE-L F1 between responses and best response: {metrics['avg_rougeL']:.3f}."
        prompt_lines.append(line)
    prompt_lines.append("Provide a concise overall summary comparing the performance of these models.")
    prompt_text = "\n".join(prompt_lines)
    if ws:
        await ws.send_json({
            "event": "progress",
            "message": f"[{current_timestamp()}] Generating overall summary using phi3:mini..."
        })
    logger.info("Summary prompt: " + prompt_text)
    summary = await generate_response_from_ollama(SUMMARY_MODEL, prompt_text, ws)
    return summary

async def process_model_report(query: str, model: str, n: int, reference: str = "", ws=None) -> (str, dict):
    """
    Generate `n` responses for the given model and query, compute metrics and visuals.
    Returns a tuple of (HTML section for this model, summary metrics dict).
    If a WebSocket (ws) is provided, sends progress updates with timestamps.
    """
    start_time = time.time()
    responses = []
    for i in range(n):
        generated_text = await generate_response_from_ollama(model, query, ws, i=i, n=n)
        responses.append(generated_text)
        if ws:
            await ws.send_json({
                "event": "progress",
                "message": f"[{current_timestamp()}] Model {model}: Generated response {i+1}/{n}."
            })
    embeddings = generate_embeddings(responses)
    sim_matrix = calculate_similarity_matrix(embeddings)
    ranked_indices, total_similarity = rank_responses_by_similarity(sim_matrix)
    stats = assess_consistency(sim_matrix, responses)
    top_resp = best_response(responses, total_similarity)
    runtime = time.time() - start_time
    timestamp = current_timestamp()
    base_name = f"{model.replace(':', '_')}_{int(time.time())}"

    emb_img_path = os.path.join(REPORTS_FOLDER, f"{base_name}_embeddings.png")
    dist_img_path = os.path.join(REPORTS_FOLDER, f"{base_name}_similarity.png")
    heat_img_path = os.path.join(REPORTS_FOLDER, f"{base_name}_heatmap.png")
    bar_img_path  = os.path.join(REPORTS_FOLDER, f"{base_name}_bar.png")
    save_embeddings_plot(embeddings, emb_img_path)
    save_similarity_distribution(sim_matrix, dist_img_path)
    save_similarity_heatmap(sim_matrix, heat_img_path)
    save_total_similarity_bar_chart(total_similarity, bar_img_path)
    emb_b64  = image_to_base64(emb_img_path) if os.path.exists(emb_img_path) else ""
    dist_b64 = image_to_base64(dist_img_path) if os.path.exists(dist_img_path) else ""
    heat_b64 = image_to_base64(heat_img_path) if os.path.exists(heat_img_path) else ""
    bar_b64  = image_to_base64(bar_img_path) if os.path.exists(bar_img_path) else ""
    for path in [emb_img_path, dist_img_path, heat_img_path, bar_img_path]:
        if os.path.exists(path):
            os.remove(path)

    # Compute BLEU and ROUGE-L scores for each response compared to the best response.
    individual_scores = {}
    for idx, response in enumerate(responses):
        bleu_score = compute_bleu(reference, response)
        rouge_score = compute_rouge_l(reference, response)
        individual_scores[idx] = {"bleu": bleu_score, "rouge": rouge_score}

    row_strs = []
    for idx, stat in stats.items():
        bleu_val = individual_scores[idx]["bleu"]
        rouge_val = individual_scores[idx]["rouge"]
        row_str = (
            f"<tr><td>{idx}</td><td>{stat['response']}</td>"
            f"<td>{stat['total_similarity']:.3f}</td>"
            f"<td>{stat['min_similarity']:.3f}</td>"
            f"<td>{stat['max_similarity']:.3f}</td>"
            f"<td>{stat['variance']:.5f}</td>"
            f"<td>{bleu_val:.3f}</td>"
            f"<td>{rouge_val:.3f}</td></tr>"
        )
        row_strs.append(row_str)
    table_html = "".join(row_strs)

    model_section_html = f"""
    <section>
      <h2>Model: {model}</h2>
      <p><strong>Generation Time:</strong> {runtime:.2f} seconds</p>
      <p><strong>Number of Responses:</strong> {n}</p>
      <p><strong>Timestamp:</strong> {timestamp}</p>
      <p><strong>Reference Answer:</strong> {reference or 'None'}</p>
      <h3>Best Response (most similar to others)</h3>
      <p>{top_resp}</p>
      <h3>Responses (and Similarity Statistics)</h3>
      <table border="1" cellspacing="0" cellpadding="5">
        <tr>
          <th>Index</th>
          <th>Response</th>
          <th>Total Sim.</th>
          <th>Min Sim.</th>
          <th>Max Sim.</th>
          <th>Variance</th>
          <th>BLEU vs Best</th>
          <th>ROUGE-L vs Best</th>
        </tr>
        {table_html}
      </table>
      <br/>
      <h4>Embeddings PCA Plot:</h4>
      {"<img src='"+emb_b64+"' alt='Embeddings PCA' />" if emb_b64 else "<p>Not available</p>"}
      <h4>Similarity Distribution:</h4>
      {"<img src='"+dist_b64+"' alt='Similarity Distribution' />" if dist_b64 else "<p>Not available</p>"}
      <h4>Similarity Matrix Heatmap:</h4>
      {"<img src='"+heat_b64+"' alt='Similarity Heatmap' />" if heat_b64 else "<p>Not available</p>"}
      <h4>Total Similarity Bar Chart:</h4>
      {"<img src='"+bar_b64+"' alt='Total Similarity Bar Chart' />" if bar_b64 else "<p>Not available</p>"}
      <br/>
      {("" if reference == "" else f"<p><strong>Avg BLEU vs Reference:</strong> {sum([compute_bleu(reference, r) for r in responses])/len(responses):.4f}</p>")}
      {("" if reference == "" else f"<p><strong>Avg ROUGE-L F1 vs Reference:</strong> {sum([compute_rouge_l(reference, r) for r in responses])/len(responses):.4f}</p>")}
    </section>
    """
    summary = {
        "model": model,
        "avg_total_similarity": float(np.mean(total_similarity)) if total_similarity.size > 0 else 0.0,
        "avg_variance": float(np.mean([stat["variance"] for stat in stats.values()])) if stats else 0.0,
        "best_total_similarity": float(np.max(total_similarity)) if total_similarity.size > 0 else 0.0,
        "avg_bleu": float(sum([individual_scores[i]["bleu"] for i in individual_scores]) / len(individual_scores)) if individual_scores else None,
        "avg_rougeL": float(sum([individual_scores[i]["rouge"] for i in individual_scores]) / len(individual_scores)) if individual_scores else None
    }
    return model_section_html, summary

async def generate_comparative_report(query: str, model_list: list, n: int, reference: str = "", ws=None) -> str:
    report_sections = ""
    summaries = []
    total_models = len(model_list)
    for idx, model in enumerate(model_list, start=1):
        if ws:
            await ws.send_json({
                "event": "progress",
                "message": f"[{current_timestamp()}] Processing model {model} ({idx}/{total_models})..."
            })
        section, summary = await process_model_report(query, model, n, reference, ws)
        report_sections += section
        summaries.append(summary)
    overall_summary = await generate_summary_comparison(summaries, ws)
    models = [s["model"] for s in summaries]
    avg_sims = [s["avg_total_similarity"] for s in summaries]
    avg_vars = [s["avg_variance"] for s in summaries]
    plt.figure(figsize=(6, 4))
    plt.bar(models, avg_sims, color='mediumseagreen')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Avg Total Similarity")
    plt.title("Average Total Similarity per Model")
    plt.tight_layout()
    sim_chart_path = os.path.join(REPORTS_FOLDER, "overall_avg_similarity.png")
    plt.savefig(sim_chart_path); plt.close()
    plt.figure(figsize=(6, 4))
    plt.bar(models, avg_vars, color='tomato')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Average Variance")
    plt.title("Average Variance per Model")
    plt.tight_layout()
    var_chart_path = os.path.join(REPORTS_FOLDER, "overall_avg_variance.png")
    plt.savefig(var_chart_path); plt.close()
    overall_sim_b64 = image_to_base64(sim_chart_path) if os.path.exists(sim_chart_path) else ""
    overall_var_b64 = image_to_base64(var_chart_path) if os.path.exists(var_chart_path) else ""
    if os.path.exists(sim_chart_path):
        os.remove(sim_chart_path)
    if os.path.exists(var_chart_path):
        os.remove(var_chart_path)
    timestamp = int(time.time())
    timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(timestamp))
    final_html = f"""
    <html>
      <head>
        <meta charset="UTF-8" />
        <title>LLM Comparative Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 20px; }}
          h1 {{ color: navy; }}
          table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
          th, td {{ border: 1px solid #999; padding: 5px; text-align: left; vertical-align: top; }}
          th {{ background: #eee; }}
          img {{ max-width: 100%; height: auto; margin: 10px 0; }}
          .summary-box {{ border: 2px solid #444; padding: 10px; background: #f9f9f9; margin-bottom: 20px; }}
          .score-explanation {{ font-size: 0.9em; color: #555; }}
        </style>
      </head>
      <body>
        <div class="summary-box">
          <h2>Overall Summary</h2>
          <p>{overall_summary}</p>
        </div>
        <h1>Comparative Report</h1>
        <p><strong>Query:</strong> {query}</p>
        <p><strong>Models:</strong> {", ".join(models)}</p>
        <p><strong>Responses per Model:</strong> {n}</p>
        <p><strong>Generated at:</strong> {current_timestamp()}</p>
        <p><strong>Reference Answer:</strong> {reference or 'None'}</p>
        <h2>Overall Comparison</h2>
        <h3>Average Total Similarity per Model</h3>
        {"<img src='"+overall_sim_b64+"' alt='Avg Similarity Chart' />" if overall_sim_b64 else "<p>Chart not available</p>"}
        <h3>Average Variance per Model</h3>
        {"<img src='"+overall_var_b64+"' alt='Avg Variance Chart' />" if overall_var_b64 else "<p>Chart not available</p>"}
        {report_sections}
        <br/>
        <p class="score-explanation">
          <strong>BLEU Score:</strong> Measures n-gram overlap between candidate and reference, indicating precision in matching phrases with the human response.
          <strong>ROUGE-L F1 Score:</strong> Reflects the longest common subsequence between candidate and reference, balancing recall and precision with the human response.
        </p>
      </body>
    </html>
    """
    report_filename = f"comparative_report_{timestamp_str}.html"
    final_path = os.path.join(REPORTS_FOLDER, report_filename)
    with open(final_path, "w") as f:
        f.write(final_html)
    logger.info(f"Comparative report generated and saved at: {final_path}")
    return report_filename
