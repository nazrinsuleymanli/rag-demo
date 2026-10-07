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
import java.util.stream.Collectors;

/**
 * Minimal Retrieval-Augmented Generation (RAG) demo with Spring AI + Ollama.
 *
 * Flow:
 *   1. Load a document (FAQ file) and split it into chunks.
 *   2. Embed the chunks and store them in an in-memory vector store.
 *   3. For a question, retrieve the most relevant chunks (vector search).
 *   4. Inject those chunks into the prompt and let the model answer from them.
 *
 * The model answers grounded on the retrieved context instead of hallucinating.
 */
@SpringBootApplication
public class RagDemoApplication {

    private static final String SOURCE_FILE = "faq.txt";
    private static final int TOP_K = 2;

    public static void main(String[] args) {
        SpringApplication.run(RagDemoApplication.class, args);
    }

    /** In-memory vector store. Spring AI auto-wires the Ollama EmbeddingModel. */
    @Bean
    VectorStore vectorStore(EmbeddingModel embeddingModel) {
        return SimpleVectorStore.builder(embeddingModel).build();
    }

    @Bean
    CommandLineRunner run(VectorStore vectorStore, ChatClient.Builder chatClientBuilder) {
        return args -> {
            ChatClient chat = chatClientBuilder.build();

            ingestDocuments(vectorStore);

            String question = "How long do refunds take?";
            String answer = answer(vectorStore, chat, question);

            System.out.println("\n--- ANSWER ---\n" + answer);
        };
    }

    /** Steps 1 & 2: read the file, split into chunks, embed and store. */
    private void ingestDocuments(VectorStore vectorStore) {
        TikaDocumentReader reader = new TikaDocumentReader(new ClassPathResource(SOURCE_FILE));
        List<Document> rawDocuments = reader.get();

        List<Document> chunks = new TokenTextSplitter().apply(rawDocuments);
        vectorStore.add(chunks);

        System.out.println("Loaded " + chunks.size() + " chunks from " + SOURCE_FILE);
    }

    /** Steps 3 & 4: retrieve relevant chunks, build the prompt, call the model. */
    private String answer(VectorStore vectorStore, ChatClient chat, String question) {
        // 3) RETRIEVE: find the most relevant chunks for this question
        List<Document> hits = vectorStore.similaritySearch(
                SearchRequest.builder().query(question).topK(TOP_K).build());

        System.out.println("\n--- RETRIEVED (what vector search found) ---");
        hits.forEach(doc -> System.out.println("* " + doc.getText()));

        // 3b) build the context from the retrieved chunks
        String context = hits.stream()
                .map(Document::getText)
                .collect(Collectors.joining("\n"));

        // 4) AUGMENT: put the context into the prompt
        String prompt = """
                Answer using ONLY the context below. If the answer is not there, say you don't know.

                Context:
                %s

                Question: %s
                """.formatted(context, question);

        System.out.println("\n--- FINAL PROMPT (what the model actually sees) ---\n" + prompt);

        // 4b) GENERATE: call the LLM
        return chat.prompt().user(prompt).call().content();
    }
}