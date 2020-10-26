# Using Gretel Cloud for Publish and Subscribe Data Labeling

This blueprint demonsrates how to send raw records and consume labeled records from Gretel's APIs.  It automatically
creates a temporary project for you with use of a context manager. You can, of course, create a project more permmanently
and write records to it and consume from them in a similar way.

When you call `start()`, the Gretel Console URL will be printed. Feel free to visit this link and observe how the
records are ingested, labeled, and explorable in our console!