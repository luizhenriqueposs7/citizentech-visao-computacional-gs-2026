function ligarAnalise() {
  const form = document.querySelector(".vision-upload");
  const painel = document.querySelector("#acompanhamento");
  const progresso = document.querySelector("#progresso");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const botao = form.querySelector("button[type=submit]");
    botao.disabled = true;
    painel.hidden = false;
    const anterior = document.querySelector('[aria-labelledby="resultado"]');
    if (anterior) anterior.hidden = true;
    const quadro = document.querySelector("#quadro-atual");
    quadro.hidden = true;
    for (const id of ["nome-quadro", "classe-atual", "confianca-atual", "revisao-atual"]) {
      document.getElementById(id).textContent = "";
    }
    progresso.textContent = "Enviando os arquivos e iniciando o modelo…";
    painel.scrollIntoView({block: "start"});
    let leitor;
    try {
      const resposta = await fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        headers: {Accept: "application/x-ndjson"},
      });
      if (!resposta.ok) {
        throw new Error(resposta.status === 413
          ? "O envio ultrapassou 24 MB."
          : "Confira os arquivos enviados e tente novamente.");
      }
      leitor = resposta.body.getReader();
      const decoder = new TextDecoder();
      let pendente = "";
      let concluido = false;
      while (true) {
        const {done, value} = await leitor.read();
        pendente += decoder.decode(value, {stream: !done});
        const linhas = pendente.split("\n");
        pendente = linhas.pop();
        for (const linha of linhas) {
          if (!linha.trim()) continue;
          const evento = JSON.parse(linha);
          if (evento.tipo === "erro") throw new Error(evento.mensagem);
          if (evento.tipo === "quadro") {
            const resultado = evento.resultado;
            progresso.textContent = "Total processado " + evento.total;
            document.querySelector("#nome-quadro").textContent = resultado.nome;
            document.querySelector("#classe-atual").textContent = resultado.classe;
            document.querySelector("#confianca-atual").textContent =
              (resultado.confianca * 100).toFixed(1) + "% de confiança";
            document.querySelector("#revisao-atual").textContent = resultado.alerta
              ? "Encaminhar para inspeção humana"
              : "Conferir a superfície em campo";
            quadro.src = "data:image/jpeg;base64," + resultado.imagem;
            quadro.hidden = false;
          }
          if (evento.tipo === "concluido") {
            const documento = new DOMParser().parseFromString(evento.html, "text/html");
            document.querySelector("main").replaceWith(documento.querySelector("main"));
            ligarAnalise();
            document.querySelector("#resultado").scrollIntoView({block: "start"});
            concluido = true;
          }
        }
        if (done) break;
      }
      if (!concluido) throw new Error("A execução foi interrompida. Tente analisar novamente.");
    } catch (erro) {
      progresso.textContent = erro.message;
      painel.scrollIntoView({block: "start"});
    } finally {
      if (leitor) await leitor.cancel().catch(() => {});
      botao.disabled = false;
    }
  });
}

ligarAnalise();
