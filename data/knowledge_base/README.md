# Knowledge Base Structure

This directory contains the knowledge base for the Instagram DSPy chatbot. It is organized into subdirectories by content type.

## Directory Structure

- `/faqs/`: Frequently asked questions and their answers
- `/instagram_content/`: Content from Instagram posts, stories, and comments
- `/travel_guides/`: Information about travel destinations, tips, and recommendations
- `/personal_info/`: Personal information for personalized responses
- `/video_transcripts/`: Transcribed content from videos

## Content Format Standards

### Markdown Files (.md)
Use markdown files for structured content with headings, lists, and basic formatting.

Example structure:
```markdown
# Title of the Content

## Metadata
- Date: YYYY-MM-DD
- Category: Primary category
- Tags: tag1, tag2, tag3
- Source: URL or description of source
- Last Updated: YYYY-MM-DD

## Content
Main content goes here...

### Subheadings as needed
More content...
```

### JSON Files (.json)
Use JSON files for structured data that requires specific schema or for metadata.

Example structure:
```json
{
  "title": "Content Title",
  "date_created": "YYYY-MM-DD",
  "last_updated": "YYYY-MM-DD",
  "category": "Primary Category",
  "tags": ["tag1", "tag2", "tag3"],
  "source": "Source information",
  "content": {
    "summary": "Brief summary of the content",
    "sections": [
      {
        "heading": "Section Heading",
        "text": "Section content..."
      }
    ]
  }
}
```

### Text Files (.txt)
Use plain text files for simple, unformatted content.

## Metadata Schema

All content should include the following metadata:

- **Title**: Descriptive title of the content
- **Date Created**: When the content was initially created (YYYY-MM-DD)
- **Last Updated**: When the content was last modified (YYYY-MM-DD)
- **Category**: Primary category for classification
- **Tags**: Multiple tags for more specific classification
- **Source**: Where the content originated from
- **Relevance Score** (optional): 1-10 rating of importance
- **Content Type**: The type of content (faq, guide, post, etc.)

## File Naming Convention

Use descriptive, kebab-case filenames:
- `topic-subtopic-specifics.extension`
- Example: `paris-attractions-eiffel-tower.md`

## Special Considerations

1. **Images**: Store image references as URLs or paths in the content files
2. **Updates**: When updating content, always update the "Last Updated" metadata
3. **Cross-references**: Use relative links to reference other knowledge base content 