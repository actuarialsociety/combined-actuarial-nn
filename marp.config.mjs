// Marp CLI config, picked up automatically from the repo root (slides.yml, `marp -s lessons`).
//
// Lessons write display math as ```math fences: GitHub and the wiki render those
// everywhere, including inside <details> panels, where GitHub's $$ support is unreliable.
// Marp only understands $$ blocks, so this rewrites the fences before parsing.
const mathFences = (md) => {
  md.core.ruler.before("normalize", "math_fences", (state) => {
    state.src = state.src.replace(
      /^([ \t]*)```math[ \t]*\n([\s\S]*?)^\1```[ \t]*$/gm,
      (_, indent, body) => `${indent}$$\n${body}${indent}$$`,
    );
  });
};

export default {
  html: true, // the Core ideas panels are <details> elements
  allowLocalFiles: true,
  engine: ({ marp }) => marp.use(mathFences),
};
