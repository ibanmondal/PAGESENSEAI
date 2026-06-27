# PageSense AI Model Evaluation Results
This document contains the automated evaluation results of the RAG pipeline testing retrieval precision, synthesis, comparison, and summarization against a pool of distractor tabs.
**Total Indexed Tabs:** 33
**Session ID:** `e70f0c0e-36f4-4aff-97b6-c4cf6df63d25`
---
## Query: What is the deepest part of the ocean and how deep is it?
- **Mode:** `ask`
- **Model Used:** `gemini-2.5-flash`
- **Confidence Score:** `0.0`

### Answer:
generate: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash\nPlease retry in 5.039487427s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '5s'}]}}

### Citations & Retrieval Quality:
- **Oceanography** (Re-ranker Score: `0.9999`)
  > *"The Mariana Trench is the deepest part of the world's oceans. It is located in the western Pacific Ocean and reaches a maximum-known depth of about 10,984 meters...."*
- **Cybersecurity** (Re-ranker Score: `0.0000`)
  > *"Computer security, cybersecurity, or information technology security is the protection of computer systems and networks from information disclosure, theft of or damage to their hardware, software, or electronic data...."*
- **Photosynthesis** (Re-ranker Score: `0.0000`)
  > *"Photosynthesis is a process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy...."*
- **Psychology** (Re-ranker Score: `0.0000`)
  > *"Psychology is the scientific study of mind and behavior. Psychology includes the study of conscious and unconscious phenomena, including feelings and thoughts...."*
- **Microbiology** (Re-ranker Score: `0.0000`)
  > *"Microbiology is the study of all living organisms that are too small to be visible with the naked eye. This includes bacteria, archaea, viruses, fungi, prions, protozoa and algae...."*

---
## Query: When did the Industrial Revolution happen?
- **Mode:** `ask`
- **Model Used:** `gemini-2.5-flash`
- **Confidence Score:** `0.0`

### Answer:
generate: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash\nPlease retry in 45.415422784s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '45s'}]}}

### Citations & Retrieval Quality:
- **The Industrial Revolution** (Re-ranker Score: `0.9999`)
  > *"The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States, in the period from about 1760 to sometime between 1820 and 1840...."*
- **The French Revolution** (Re-ranker Score: `0.0744`)
  > *"The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789 and ended with the formation of the French Consulate in 1799...."*
- **World War II** (Re-ranker Score: `0.0003`)
  > *"World War II was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries forming two opposing military alliances: the Allies and the Axis powers...."*
- **History of Rome** (Re-ranker Score: `0.0002`)
  > *"The Roman Empire was founded in 27 BC by Augustus. It became one of the largest empires in the ancient world, encompassing most of Europe, North Africa, and the Middle East...."*
- **History of the Internet - Wikipedia** (Re-ranker Score: `0.0002`)
  > *"The history of the Internet has its origin in the efforts of scientists and engineers to build and interconnect computer networks. The Internet Protocol Suite, the set of rules used to communicate between networks and devices on the Internet, arose from research and development i..."*

---
## Query: Explain what a blockchain is.
- **Mode:** `ask`
- **Model Used:** `gemini-2.5-flash`
- **Confidence Score:** `0.64`

### Answer:
A blockchain is a growing list of records, called blocks, that are securely linked together using cryptography [#1]. Each block contains a cryptographic hash of the previous block [#1]. Cryptography is the practice and study of techniques for secure communication in the presence of adversarial behavior, generally involving constructing and analyzing protocols that prevent third parties or the public from reading private messages [#2].

### Citations & Retrieval Quality:
- **Blockchain Technology** (Re-ranker Score: `0.9997`)
  > *"A blockchain is a growing list of records, called blocks, that are securely linked together using cryptography. Each block contains a cryptographic hash of the previous block...."*
- **Cryptography** (Re-ranker Score: `0.0008`)
  > *"Cryptography, or cryptology, is the practice and study of techniques for secure communication in the presence of adversarial behavior. More generally, cryptography is about constructing and analyzing protocols that prevent third parties or the public from reading private messages..."*
- **Cybersecurity** (Re-ranker Score: `0.0001`)
  > *"Computer security, cybersecurity, or information technology security is the protection of computer systems and networks from information disclosure, theft of or damage to their hardware, software, or electronic data...."*
- **Genetics** (Re-ranker Score: `0.0001`)
  > *"DNA is a polymer composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions for the development, functioning, growth and reproduction of all known organisms...."*
- **Artificial Neural Networks** (Re-ranker Score: `0.0001`)
  > *"Artificial neural networks are computing systems inspired by the biological neural networks that constitute animal brains. They 'learn' to perform tasks by considering examples...."*

---
## Query: How do Python and JavaScript differ in their design and typing?
- **Mode:** `compare`
- **Model Used:** `gemini-2.5-flash`
- **Confidence Score:** `0.0`

### Answer:
generate: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash\nPlease retry in 3.601618003s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '3s'}]}}

### Citations & Retrieval Quality:
- **Python - Wikipedia** (Re-ranker Score: `0.9154`)
  > *"Python is a high-level, interpreted, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structure..."*
- **Python Programming** (Re-ranker Score: `0.7980`)
  > *"Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation...."*
- **JavaScript - Wikipedia** (Re-ranker Score: `0.5769`)
  > *"JavaScript, often abbreviated as JS, is a programming language that is one of the core technologies of the World Wide Web, alongside HTML and CSS. As of 2023, 98.7% of websites use JavaScript on the client side for webpage behavior. JavaScript is a high-level, often just-in-time ..."*
- **Robotics** (Re-ranker Score: `0.0000`)
  > *"Robotics is an interdisciplinary branch of computer science and engineering. Robotics involves design, construction, operation, and use of robots. The goal of robotics is to design machines that can help and assist humans...."*
- **Cryptography** (Re-ranker Score: `0.0000`)
  > *"Cryptography, or cryptology, is the practice and study of techniques for secure communication in the presence of adversarial behavior. More generally, cryptography is about constructing and analyzing protocols that prevent third parties or the public from reading private messages..."*

---
## Query: Summarize the history of the internet.
- **Mode:** `summarize`
- **Model Used:** `gemini-2.5-flash`
- **Confidence Score:** `0.0`

### Answer:
generate: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash\nPlease retry in 45.001140582s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash', 'location': 'global'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '45s'}]}}

### Citations & Retrieval Quality:
- **History of the Internet - Wikipedia** (Re-ranker Score: `0.9982`)
  > *"The history of the Internet has its origin in the efforts of scientists and engineers to build and interconnect computer networks. The Internet Protocol Suite, the set of rules used to communicate between networks and devices on the Internet, arose from research and development i..."*
- **The Renaissance** (Re-ranker Score: `0.0001`)
  > *"The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries...."*
- **The Industrial Revolution** (Re-ranker Score: `0.0001`)
  > *"The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States, in the period from about 1760 to sometime between 1820 and 1840...."*
- **History of Rome** (Re-ranker Score: `0.0000`)
  > *"The Roman Empire was founded in 27 BC by Augustus. It became one of the largest empires in the ancient world, encompassing most of Europe, North Africa, and the Middle East...."*
- **Cybersecurity** (Re-ranker Score: `0.0000`)
  > *"Computer security, cybersecurity, or information technology security is the protection of computer systems and networks from information disclosure, theft of or damage to their hardware, software, or electronic data...."*

---
