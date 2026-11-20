"""R-092: the module contract of the modular arm, checked on source text alone (AST; nothing is executed).

A modular ``GeneratedPolicy`` declares a class-level literal ``MODULE_SPEC`` naming at least two of its own methods as
modules, each with the observation fields it reads (``inputs``), what it returns (``outputs``, ``units``) and the other
modules whose outputs it receives (``dependencies``). :func:`check_module_contract` holds the code to that declaration:

* every observation read (``self.obs_field(obs, "<name>")``) is inside a declared module, and each module's set of reads
  equals its ``inputs``;
* ``forward`` never reads the observation: it may hand ``obs`` to a declared module and look at ``obs.shape`` /
  ``obs.device`` / ``obs.dtype``, nothing else. Any other use is an undeclared observation-to-action bypass. The same
  holds for the observation inside a module, where ``self.obs_field`` is the one further use;
* every declared module is called from ``forward``, and the modules whose outputs reach a module (as arguments in
  ``forward``, or by a direct ``self.<module>(...)`` call inside it) equal its ``dependencies``.

``forward`` is otherwise free: arithmetic on module outputs, parameters and constants is the generated composition.
The choice of modules is the LLM's; nothing here names a role. A defect is one line the LLM can act on; the modular
arm's generator treats a non-empty list as a validation failure (stage ``module_contract``) that gets the slot's one
linked correction. The monolithic arm never sees this module.

Known gap: a value stored on ``self`` in ``forward`` and read inside a module is not followed.
"""

from __future__ import annotations

import ast

CLASS_NAME = "GeneratedPolicy"
SPEC_NAME = "MODULE_SPEC"
SPEC_KEYS = ("inputs", "outputs", "units", "dependencies")
RESERVED = ("forward", "__init__", "get_param_ranges")
OBS_ATTRIBUTES = ("shape", "device", "dtype", "size", "ndim", "dim")
MIN_MODULES = 2

MODULAR_REQUIREMENT = """
Structural requirement for this controller: it must be written as named modules with a declared interface, and the
declaration is checked against the code before the controller is run.

1. Declare a class attribute `MODULE_SPEC` in the body of `GeneratedPolicy`, written as a plain Python literal (only
   dicts, lists and strings; no variables, no function calls, no comprehensions):

       MODULE_SPEC = {
           "<method name>": {
               "inputs": ["<observation field name>", ...],
               "outputs": ["<name of a returned quantity>", ...],
               "units": {"<output name>": "<unit or meaning>", ...},
               "dependencies": ["<name of another module>", ...],
           },
           ...
       }

   Each key is the name of a method of `GeneratedPolicy`; that method is a module. All four entries are required for
   every module. `inputs` and `dependencies` may be empty lists; `outputs` must name at least one quantity.
2. Declare at least two modules. How the controller is divided into modules, how many there are, what each computes and
   what they are called is your own design decision. `forward`, `__init__` and `get_param_ranges` cannot be modules.
3. Observation reads belong to modules. Every `self.obs_field(obs, "<name>")` call must be inside a module, with the
   field name written as a string literal, and the set of fields a module reads must be exactly its `inputs` list
   (nothing read that is not listed, nothing listed that is not read). A method that is not a declared module may not
   call `self.obs_field`.
4. `forward(self, obs)` never reads the observation itself. Inside `forward`, `obs` may appear only (a) as an argument
   passed directly to a module, e.g. `self.<module>(obs)`, or (b) as `obs.shape`, `obs.size(...)`, `obs.ndim`,
   `obs.dim()`, `obs.device` or `obs.dtype`. Do not
   call `self.obs_field` in `forward`, do not assign `obs` to another name or reassign it, do not pass it to a torch
   function or to a method that is not a module, and do not do arithmetic on it. Inside a module the same holds, with
   one more allowed use: `self.obs_field(obs, "<name>")`.
5. `forward` calls every declared module directly at least once and composes their outputs into the action mean. The
   composition is yours: arithmetic on module outputs, parameters and constants is unrestricted.
6. `dependencies` must be exact. Module B depends on module A when a value returned by A (or computed in `forward`
   from a value returned by A) is passed to B as an argument, or when B calls `self.A(...)` itself. List exactly those
   modules for B: no more, no fewer. Dependencies are direct only: the value a module call returns counts as that
   module's output alone, so if C receives B's output and B received A's, C depends on B and not on A. A module that
   receives only `obs` has `"dependencies": []`.
7. Module methods are ordinary instance methods (`def name(self, ...)`) defined inside the class. Everything must be
   inside the class: code outside `class GeneratedPolicy` is discarded.
8. Every other rule in this prompt is unchanged: parameters declared in `__init__` with ranges in
   `get_param_ranges()`, row-wise computation, `forward` returning `(mean, self.fixed_std(mean))`, and so on.
9. The code skeleton and the reference example elsewhere in this prompt read observation fields inside `forward`.
   Follow their conventions for everything else, but here those reads must be inside modules, as items 3 and 4 say."""


# ---------------------------------------------------------------------------------------------------------------------
#  AST helpers
# ---------------------------------------------------------------------------------------------------------------------


def _self_call(node) -> str | None:
    """``name`` when ``node`` is the call ``self.<name>(...)``, else ``None``."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"):
        return node.func.attr
    return None


def _parents(func) -> dict:
    return {child: parent for parent in ast.walk(func) for child in ast.iter_child_nodes(parent)}


def _params(func) -> list[str]:
    """The method's parameter names after ``self``, positional order first."""
    a = func.args
    return [x.arg for x in (a.posonlyargs + a.args)][1:] + [x.arg for x in a.kwonlyargs]


def _call_args(call) -> list[tuple]:
    """``[(position or keyword, expression)]`` of a call's arguments (starred ones under ``None``)."""
    out = [(None if isinstance(a, ast.Starred) else i, a) for i, a in enumerate(call.args)]
    return out + [(k.arg, k.value) for k in call.keywords]


def _bound_param(func, key) -> str | None:
    """The parameter of ``func`` an argument at position / keyword ``key`` binds to."""
    names = _params(func)
    if isinstance(key, int):
        positional = [x.arg for x in (func.args.posonlyargs + func.args.args)][1:]
        return positional[key] if key < len(positional) else None
    return key if key in names else None


def _field_reads(func) -> tuple[set, list]:
    """``({literal field names read}, [defects])`` of the ``self.obs_field`` uses in one method."""
    parents, reads, problems = _parents(func), set(), []
    for node in ast.walk(func):
        if not (isinstance(node, ast.Attribute) and node.attr == "obs_field"):
            continue
        call = parents.get(node)
        if not (isinstance(call, ast.Call) and call.func is node):
            problems.append(f"line {node.lineno}: `{func.name}` uses `obs_field` without calling it directly; "
                            "write `self.obs_field(obs, \"<name>\")`")
            continue
        name = call.args[1] if len(call.args) > 1 else next((k.value for k in call.keywords if k.arg == "name"), None)
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            reads.add(name.value)
        else:
            problems.append(f"line {call.lineno}: `{func.name}` calls `self.obs_field` with a field name that is not "
                            "a string literal; the declared inputs cannot be checked")
    return reads, problems


def _obs_field_seeds(func) -> set:
    """Parameters a method passes to ``self.obs_field`` as the observation."""
    out = set()
    for node in ast.walk(func):
        if _self_call(node) == "obs_field":
            first = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "obs"), None)
            if isinstance(first, ast.Name) and first.id in _params(func):
                out.add(first.id)
    return out


def _observation_uses(func, obs_names: set, modules: dict, in_forward: bool) -> tuple[list, list]:
    """Every use of the observation in one method: ``([defects], [(callee module, parameter receiving obs)])``."""
    parents, problems, passed = _parents(func), [], []
    where = "`forward`" if in_forward else f"module `{func.name}`"
    allowed = ("as an argument to a declared module or as obs.shape / obs.device / obs.dtype" if in_forward else
               "in self.obs_field(obs, \"<name>\"), as an argument to a declared module or as obs.shape / obs.device "
               "/ obs.dtype")
    for node in ast.walk(func):
        if not (isinstance(node, ast.Name) and node.id in obs_names):
            continue
        if not isinstance(node.ctx, ast.Load):
            problems.append(f"line {node.lineno}: {where} reassigns the observation `{node.id}`; leave it unchanged")
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.attr in OBS_ATTRIBUTES:
            continue
        call = parents.get(parent) if isinstance(parent, ast.keyword) else parent
        callee = _self_call(call)
        key = next((k for k, a in _call_args(call) if a is node), None) if callee is not None else None
        if callee == "obs_field" and key in (0, "obs"):
            continue  # a read; `forward` calling obs_field at all is reported once by the caller
        if callee in modules and key is not None:
            bound = _bound_param(modules[callee], key)
            if bound is not None:
                passed.append((callee, bound))
            continue
        problems.append(f"line {node.lineno}: undeclared observation-to-action bypass: {where} uses the observation "
                        f"`{node.id}` outside a declared module's inputs (`{ast.unparse(_statement(node, parents))}`)"
                        f"; it may appear only {allowed}")
    return problems, passed


def _statement(node, parents):
    """The innermost expression worth quoting around ``node``: its parent call or attribute, else itself."""
    parent = parents.get(node)
    if isinstance(parent, ast.keyword):
        parent = parents.get(parent)
    return parent if isinstance(parent, (ast.Call, ast.Attribute, ast.BinOp, ast.Subscript, ast.Assign)) else node


def _spec(cls) -> tuple[dict | None, list]:
    """The ``MODULE_SPEC`` literal of the class and the defects of its shape."""
    value = None
    for stmt in cls.body:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == SPEC_NAME for t in targets):
            value = stmt.value
    if value is None:
        return None, [f"`{CLASS_NAME}` has no class-level `{SPEC_NAME}`; declare it in the class body as a literal "
                      "dict {method_name: {\"inputs\": [...], \"outputs\": [...], \"units\": {...}, "
                      "\"dependencies\": [...]}}"]
    try:
        spec = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return None, [f"`{SPEC_NAME}` is not a pure literal; write it with dicts, lists and strings only (no "
                      "variables, calls or comprehensions)"]
    if not isinstance(spec, dict) or not all(isinstance(k, str) for k in spec):
        return None, [f"`{SPEC_NAME}` must be a dict keyed by module method names"]
    problems = []
    if len(spec) < MIN_MODULES:
        problems.append(f"`{SPEC_NAME}` declares {len(spec)} module(s); at least {MIN_MODULES} are required")
    for name, entry in spec.items():
        if not isinstance(entry, dict):
            problems.append(f"`{SPEC_NAME}[\"{name}\"]` must be a dict with the keys {list(SPEC_KEYS)}")
            spec[name] = {}
            continue
        missing = [k for k in SPEC_KEYS if k not in entry]
        if missing:
            problems.append(f"`{SPEC_NAME}[\"{name}\"]` is missing {missing}")
        for key in ("inputs", "outputs", "dependencies"):
            v = entry.get(key, [])
            if not isinstance(v, (list, tuple)) or not all(isinstance(x, str) for x in v):
                problems.append(f"`{SPEC_NAME}[\"{name}\"][\"{key}\"]` must be a list of strings")
                entry[key] = []
        if "outputs" in entry and not entry["outputs"]:
            problems.append(f"`{SPEC_NAME}[\"{name}\"][\"outputs\"]` is empty; name at least one returned quantity")
        if "units" in entry and not isinstance(entry["units"], (dict, str)):
            problems.append(f"`{SPEC_NAME}[\"{name}\"][\"units\"]` must be a dict (output name -> unit) or a string")
        for dep in entry.get("dependencies", []):
            if dep == name:
                problems.append(f"module `{name}` lists itself in its dependencies")
            elif dep not in spec:
                problems.append(f"module `{name}` lists the dependency `{dep}`, which is not a declared module")
    return spec, problems


# ---------------------------------------------------------------------------------------------------------------------
#  Dataflow in forward
# ---------------------------------------------------------------------------------------------------------------------


class _Flow:
    """Which modules' outputs each local name of ``forward`` carries, and which modules each module call receives.

    A ``self.<module>(...)`` call is tagged with that module alone; any other expression carries the union of the tags
    in it. Statements are followed in source order; a top-level assignment replaces a name's tags, one inside a
    compound statement (a branch or loop, walked twice) adds to them.
    """

    def __init__(self, modules: dict):
        self.modules = modules
        self.tags: dict[str, set] = {}
        self.called: set = set()
        self.received: dict[str, set] = {m: set() for m in modules}

    def expr(self, node) -> set:
        if node is None:
            return set()
        callee = _self_call(node)
        if callee in self.modules:
            self.called.add(callee)
            for _, arg in _call_args(node):
                self.received[callee] |= self.expr(arg.value if isinstance(arg, ast.Starred) else arg) - {callee}
            return {callee}
        if isinstance(node, ast.Name):
            return set(self.tags.get(node.id, ()))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            return set(self.tags.get(f"self.{node.attr}", ()))
        out = set()
        for child in ast.iter_child_nodes(node):
            out |= self.expr(child)
        return out

    def assign(self, target, tags: set, replace: bool) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for t in target.elts:
                self.assign(t, tags, replace)
            return
        if isinstance(target, ast.Starred):
            return self.assign(target.value, tags, replace)
        if isinstance(target, ast.Subscript):  # action[:, 0] = x adds to what `action` carries
            tags, replace = tags | self.expr(target.slice), False
            target = target.value
        if isinstance(target, ast.Name):
            key = target.id
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
            key = f"self.{target.attr}"
        else:
            return
        self.tags[key] = set(tags) if replace else self.tags.get(key, set()) | tags

    def block(self, stmts, top: bool) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                tags = self.expr(stmt.value)
                for t in stmt.targets:
                    self.assign(t, tags, replace=top)
            elif isinstance(stmt, ast.AnnAssign):
                self.assign(stmt.target, self.expr(stmt.value), replace=top)
            elif isinstance(stmt, ast.AugAssign):
                self.assign(stmt.target, self.expr(stmt.value), replace=False)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.block(stmt.body, top=False)
            else:
                bodies = [getattr(stmt, f) for f in ("body", "orelse", "finalbody") if getattr(stmt, f, None)]
                bodies += [h.body for h in getattr(stmt, "handlers", [])]
                for field, value in ast.iter_fields(stmt):
                    if field not in ("body", "orelse", "finalbody", "handlers"):
                        for node in (value if isinstance(value, list) else [value]):
                            if isinstance(node, ast.withitem):
                                tags = self.expr(node.context_expr)
                                if node.optional_vars is not None:
                                    self.assign(node.optional_vars, tags, replace=False)
                            elif isinstance(node, ast.AST):
                                tags = self.expr(node)
                                if isinstance(stmt, (ast.For, ast.AsyncFor)) and field == "iter":
                                    self.assign(stmt.target, tags, replace=False)
                for _ in range(2 if bodies else 0):  # twice, so a loop-carried value reaches the calls above it
                    for body in bodies:
                        self.block(body, top=False)


# ---------------------------------------------------------------------------------------------------------------------
#  The contract
# ---------------------------------------------------------------------------------------------------------------------


def check_module_contract(source: str) -> list[str]:
    """Defects of ``source`` against the module contract; an empty list passes. Pure AST: nothing is executed."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"the code does not parse: {exc.msg} (line {exc.lineno})"]
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == CLASS_NAME), None)
    if cls is None:
        return [f"the code contains no class {CLASS_NAME}"]
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    spec, problems = _spec(cls)
    if spec is None:
        return problems
    modules = {}
    for name in spec:
        if name in RESERVED:
            problems.append(f"`{name}` cannot be a module; `forward`, `__init__` and `get_param_ranges` are not modules")
        elif name not in methods:
            problems.append(f"`{SPEC_NAME}` declares the module `{name}`, but `{CLASS_NAME}` has no method of that name")
        elif methods[name].decorator_list or not (methods[name].args.posonlyargs + methods[name].args.args) \
                or (methods[name].args.posonlyargs + methods[name].args.args)[0].arg != "self":
            problems.append(f"module `{name}` must be an ordinary instance method `def {name}(self, ...)` without "
                            "decorators")
        else:
            modules[name] = methods[name]
    forward = methods.get("forward")
    if forward is None:
        return problems + [f"`{CLASS_NAME}` has no `forward` method"]

    # -- observation reads: only in modules, and equal to the declared inputs -----------------------------------------
    for name, func in methods.items():
        reads, bad = _field_reads(func)
        if name in modules:
            problems.extend(bad)
            declared = set(spec[name].get("inputs", []))
            if reads != declared:
                problems.append(f"module `{name}` reads the observation fields {sorted(reads)} but declares the inputs "
                                f"{sorted(declared)}; make the two sets equal")
        elif name == "forward":
            if reads or bad:
                problems.append("`forward` calls `self.obs_field`; `forward` may not read the observation: move every "
                                "read into a declared module that lists the field in its inputs")
        elif reads or bad:
            problems.append(f"`{name}` reads the observation (`self.obs_field`) but is not a declared module; declare "
                            f"it in `{SPEC_NAME}` or move its reads into a declared module")

    # -- how the observation is used: forward first, then every module it reaches -------------------------------------
    fparams = _params(forward)
    if not fparams:
        return problems + ["`forward` must take the observation as its argument: `def forward(self, obs)`"]
    obs_params = {name: _obs_field_seeds(func) for name, func in modules.items()}
    bad, passed = _observation_uses(forward, {fparams[0]}, modules, in_forward=True)
    problems.extend(bad)
    queue, internal = list(passed), {}
    while True:
        for callee, param in queue:
            obs_params[callee].add(param)
        queue = []
        for name, func in modules.items():
            if internal.get(name, (None,))[0] == obs_params[name]:
                continue
            bad, passed = _observation_uses(func, obs_params[name], modules, in_forward=False)
            internal[name] = (set(obs_params[name]), bad)
            queue.extend((c, p) for c, p in passed if p not in obs_params[c])
        if not queue:
            break
    for name in modules:
        problems.extend(internal[name][1])

    # -- calls and dependencies -------------------------------------------------------------------------------------
    flow = _Flow(modules)
    flow.block(forward.body, top=True)
    for name, func in modules.items():
        if name not in flow.called:
            problems.append(f"module `{name}` is declared but never called from `forward`; `forward` must call every "
                            "declared module directly")
        direct = {c for c in (_self_call(n) for n in ast.walk(func)) if c in modules and c != name}
        observed, declared = flow.received[name] | direct, set(spec[name].get("dependencies", []))
        if observed != declared:
            problems.append(f"module `{name}` declares the dependencies {sorted(declared)} but receives the outputs of "
                            f"{sorted(observed)} (arguments passed to it in `forward`, plus modules it calls itself); "
                            "make the two sets equal")
    return list(dict.fromkeys(problems))


__all__ = ["MODULAR_REQUIREMENT", "MIN_MODULES", "OBS_ATTRIBUTES", "RESERVED", "SPEC_KEYS", "SPEC_NAME",
           "check_module_contract"]
