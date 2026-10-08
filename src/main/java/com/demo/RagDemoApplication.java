package com.demo;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.ClassPathResource;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * Multi-query (agentic) RAG demo — orchestrator / workers / aggregator.
 *
 *   1. ORCHESTRATOR: an LLM call splits a complex question into sub-questions.
 *   2. WORKERS      : each sub-question is answered with RAG, in parallel (CompletableFuture).
 *   3. AGGREGATOR   : an LLM call combines the sub-answers into one final answer.
 */
@SpringBootApplication
public class RagDemoApplication {

    private static final String SOURCE_FILE = "faq.txt";
    private static final int TOP_K = 2;

    public static void main(String[] args) {
        SpringApplication.run(RagDemoApplication.class, args);
    }

    @Bean
    VectorStore vectorStore(EmbeddingModel embeddingModel) {
        return SimpleVectorStore.builder(embeddingModel).build();
    }

    @Bean
    CommandLineRunner run(VectorStore vectorStore, ChatClient.Builder builder) {
        return args -> {
            ChatClient chat = builder.build();
            ingest(vectorStore);

            String question =
                    "What is your refund policy, how long does shipping take, and what payment methods do you accept?";

            long start = System.currentTimeMillis();

            // 1) ORCHESTRATOR: split the question into standalone sub-questions
            List<String> subQuestions = decompose(chat, question);
            System.out.println("\n--- DECOMPOSED into " + subQuestions.size() + " sub-questions ---");
            subQuestions.forEach(sq -> System.out.println("* " + sq));

            // 2) WORKERS (parallel): answer each sub-question with RAG
            ExecutorService pool = Executors.newFixedThreadPool(subQuestions.size());
            List<CompletableFuture<String>> futures = subQuestions.stream()
                    .map(sq -> CompletableFuture.supplyAsync(() -> answerOne(vectorStore, chat, sq), pool))
                    .toList();

            List<String> subAnswers = futures.stream()
                    .map(CompletableFuture::join)   // wait for all workers to finish
                    .toList();
            pool.shutdown();

            // 3) AGGREGATOR: combine the sub-answers into one final answer
            String finalAnswer = combine(chat, question, subQuestions, subAnswers);

            long took = System.currentTimeMillis() - start;
            System.out.println("\n--- FINAL ANSWER (" + took + " ms) ---\n" + finalAnswer);
        };
    }

    /** Read the FAQ, split into chunks, embed and store. */
    private void ingest(VectorStore vectorStore) {
        var reader = new TikaDocumentReader(new ClassPathResource(SOURCE_FILE));
        List<Document> chunks = new TokenTextSplitter().apply(reader.get());
        vectorStore.add(chunks);
        System.out.println("Loaded " + chunks.size() + " chunks from " + SOURCE_FILE);
    }

    /** ORCHESTRATOR: ask the model to break the question into standalone sub-questions. */
    private List<String> decompose(ChatClient chat, String question) {
        String prompt = """
            Break the user's question into simple, standalone sub-questions, one per line.
            If it is already simple, return just that one line.
            Return ONLY the sub-questions - no numbering, no extra text.

            Question: %s
            """.formatted(question);

        String raw = chat.prompt().user(prompt).call().content();
        return raw.lines()
                .map(String::trim)
                .filter(line -> !line.isBlank())
                .toList();
    }

    /** WORKER: standard RAG for one sub-question (retrieve + generate). */
    private String answerOne(VectorStore vectorStore, ChatClient chat, String question) {
        List<Document> hits = vectorStore.similaritySearch(
                SearchRequest.builder().query(question).topK(TOP_K).build());

        String context = hits.stream()
                .map(Document::getText)
                .collect(Collectors.joining("\n"));

        String prompt = """
            Answer using ONLY the context below. If the answer is not there, say you don't know.
            Context:
            %s
            Question: %s
            """.formatted(context, question);

        String answer = chat.prompt().user(prompt).call().content();
        System.out.println("  [worker on " + Thread.currentThread().getName() + "] "
                + question + " -> " + answer.replaceAll("\\s+", " ").trim());
        return answer;
    }

    /** AGGREGATOR: combine the sub-answers into one final answer to the original question. */
    private String combine(ChatClient chat, String originalQuestion,
                           List<String> subQuestions, List<String> subAnswers) {
        StringBuilder pairs = new StringBuilder();
        for (int i = 0; i < subQuestions.size(); i++) {
            pairs.append("Q: ").append(subQuestions.get(i)).append("\n")
                    .append("A: ").append(subAnswers.get(i)).append("\n\n");
        }

        String prompt = """
            Using the sub-answers below, write one clear, complete answer to the user's original question.

            Original question: %s

            Sub-answers:
            %s
            """.formatted(originalQuestion, pairs);

        return chat.prompt().user(prompt).call().content();
    }
}