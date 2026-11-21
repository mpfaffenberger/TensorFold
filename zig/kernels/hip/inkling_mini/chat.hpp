#pragma once
#include <string>
#include <vector>

// A chat request as the checkpoint's template sees it. Shared by the request
// parser and the tokenizer so neither depends on the other.
struct ChatPart {
    bool image = false;
    std::string text;
    size_t rows = 0; // projected image rows this placeholder expands to
};
struct ChatMessage {
    std::string role;      // system, user or assistant
    std::string reasoning; // an earlier assistant turn's reasoning_content
    std::vector<ChatPart> parts;
    bool listed = false;   // content was a part list, not one string
};
