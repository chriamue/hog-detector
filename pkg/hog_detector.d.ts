/* tslint:disable */
/* eslint-disable */
/**
* @param {AnnotatedImagesJS} images
* @returns {Promise<void>}
*/
export function init_images(images: AnnotatedImagesJS): Promise<void>;
/**
* @param {Element} root
* @param {AnnotatedImagesJS} images
*/
export function init_annotation_tool(root: Element, images: AnnotatedImagesJS): void;
/**
* @param {Element} root
* @param {AnnotatedImagesJS} images
* @param {HogDetectorJS} detector
*/
export function init_trainer(root: Element, images: AnnotatedImagesJS, detector: HogDetectorJS): void;
/**
* Handler for `console.log` invocations.
*
* If a test is currently running it takes the `args` array and stringifies
* it and appends it to the current output of the test. Otherwise it passes
* the arguments to the original `console.log` function, psased as
* `original`.
* @param {Array<any>} args
*/
export function __wbgtest_console_log(args: Array<any>): void;
/**
* Handler for `console.debug` invocations. See above.
* @param {Array<any>} args
*/
export function __wbgtest_console_debug(args: Array<any>): void;
/**
* Handler for `console.info` invocations. See above.
* @param {Array<any>} args
*/
export function __wbgtest_console_info(args: Array<any>): void;
/**
* Handler for `console.warn` invocations. See above.
* @param {Array<any>} args
*/
export function __wbgtest_console_warn(args: Array<any>): void;
/**
* Handler for `console.error` invocations. See above.
* @param {Array<any>} args
*/
export function __wbgtest_console_error(args: Array<any>): void;
/**
*/
export class AnnotatedImagesJS {
  free(): void;
/**
*/
  constructor();
}
/**
*/
export class AnnotationsJS {
  free(): void;
/**
*/
  constructor();
}
/**
*/
export class HogDetectorJS {
  free(): void;
/**
*/
  constructor();
/**
*/
  init_random_forest_classifier(): void;
/**
*/
  init_bayes_classifier(): void;
/**
*/
  init_combined_classifier(): void;
/**
* @param {Uint8Array} img_data
* @returns {Uint8Array}
*/
  next(img_data: Uint8Array): Uint8Array;
}
/**
* Runtime test harness support instantiated in JS.
*
* The node.js entry script instantiates a `Context` here which is used to
* drive test execution.
*/
export class WasmBindgenTestContext {
  free(): void;
/**
* Creates a new context ready to run tests.
*
* A `Context` is the main structure through which test execution is
* coordinated, and this will collect output and results for all executed
* tests.
*/
  constructor();
/**
* Inform this context about runtime arguments passed to the test
* harness.
*
* Eventually this will be used to support flags, but for now it's just
* used to support test filters.
* @param {any[]} args
*/
  args(args: any[]): void;
/**
* Executes a list of tests, returning a promise representing their
* eventual completion.
*
* This is the main entry point for executing tests. All the tests passed
* in are the JS `Function` object that was plucked off the
* `WebAssembly.Instance` exports list.
*
* The promise returned resolves to either `true` if all tests passed or
* `false` if at least one test failed.
* @param {any[]} tests
* @returns {Promise<any>}
*/
  run(tests: any[]): Promise<any>;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_annotationsjs_free: (a: number) => void;
  readonly annotationsjs_new: () => number;
  readonly init_images: (a: number) => number;
  readonly init_annotation_tool: (a: number, b: number) => void;
  readonly init_trainer: (a: number, b: number, c: number) => void;
  readonly __wbg_hogdetectorjs_free: (a: number) => void;
  readonly hogdetectorjs_new: () => number;
  readonly hogdetectorjs_init_random_forest_classifier: (a: number) => void;
  readonly hogdetectorjs_init_bayes_classifier: (a: number) => void;
  readonly hogdetectorjs_init_combined_classifier: (a: number) => void;
  readonly hogdetectorjs_next: (a: number, b: number, c: number, d: number) => void;
  readonly __wbg_annotatedimagesjs_free: (a: number) => void;
  readonly annotatedimagesjs_new: () => number;
  readonly __wbg_wasmbindgentestcontext_free: (a: number) => void;
  readonly wasmbindgentestcontext_new: () => number;
  readonly wasmbindgentestcontext_args: (a: number, b: number, c: number) => void;
  readonly wasmbindgentestcontext_run: (a: number, b: number, c: number) => number;
  readonly __wbgtest_console_log: (a: number) => void;
  readonly __wbgtest_console_debug: (a: number) => void;
  readonly __wbgtest_console_info: (a: number) => void;
  readonly __wbgtest_console_warn: (a: number) => void;
  readonly __wbgtest_console_error: (a: number) => void;
  readonly __wbindgen_malloc: (a: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number) => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly _dyn_core__ops__function__Fn__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__h5c6c9f5e1b935007: (a: number, b: number, c: number) => void;
  readonly _dyn_core__ops__function__FnMut__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__h03ee0a4aa64ad4f9: (a: number, b: number, c: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly wasm_bindgen__convert__closures__invoke3_mut__h0f4e99a9d37b58df: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly wasm_bindgen__convert__closures__invoke2_mut__h67f2ced87faad706: (a: number, b: number, c: number, d: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
