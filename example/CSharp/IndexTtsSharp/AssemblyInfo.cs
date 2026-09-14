using System.Runtime.CompilerServices;

// The example CLI (IndexTts.Net) hosts the numeric-validation modes, which reach into
// model internals — keep them visible to it without widening the public API.
[assembly: InternalsVisibleTo("IndexTts.Net")]
