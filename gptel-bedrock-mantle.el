;;; gptel-bedrock-mantle.el --- AWS Bedrock (bedrock-mantle) support for gptel  -*- lexical-binding: t; -*-

;; Keywords: comm, convenience

;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.

;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with this program.  If not, see <https://www.gnu.org/licenses/>.

;;; Commentary:

;; This file adds support for the AWS Bedrock `bedrock-mantle' endpoint to
;; gptel.  Unlike the `bedrock-runtime' endpoint (handled by `gptel-bedrock'),
;; which speaks the Bedrock-native Converse/InvokeModel protocol, the
;; `bedrock-mantle' endpoint exposes OpenAI-compatible APIs.  This backend uses
;; its OpenAI-compatible Responses API, so it inherits all of the request,
;; response, streaming and tool-use machinery from `gptel-openai-responses'.
;;
;; The endpoint URL has the form:
;;   https://bedrock-mantle.{region}.api.aws/openai/v1/responses
;;
;; Documentation:
;; * https://docs.aws.amazon.com/bedrock/latest/userguide/inference-endpoints.html
;;
;; Authentication uses either a Bedrock API key (a bearer token, which also
;; works with the OpenAI SDK) or AWS SigV4 request signing.  The credential
;; helpers are shared with `gptel-bedrock'.

;;; Code:
(require 'cl-lib)
(eval-and-compile (require 'gptel-openai-responses))
(require 'gptel-bedrock)

;; bedrock-mantle speaks the OpenAI Responses API, so we inherit from
;; `gptel-openai-responses' to reuse its request/response/stream/tool methods.
(cl-defstruct (gptel-bedrock-mantle (:constructor gptel--make-bedrock-mantle)
                                    (:copier nil)
                                    (:include gptel-openai-responses))
  model-region)

(defvar gptel-bedrock-mantle--model-ids
  ;; Bedrock-mantle model IDs carry a provider prefix (e.g. "openai.gpt-5.5").
  ;; The OpenAI models on mantle do not support geographic or global
  ;; cross-region inference profiles, so no region prefix is applied.
  ;; https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
  '((gpt-5.5                . "openai.gpt-5.5")
    (gpt-5.4                . "openai.gpt-5.4")
    (gpt-oss-120b           . "openai.gpt-oss-120b")
    (gpt-oss-20b            . "openai.gpt-oss-20b")
    (gpt-oss-safeguard-120b . "openai.gpt-oss-safeguard-120b")
    (gpt-oss-safeguard-20b  . "openai.gpt-oss-safeguard-20b"))
  "Map of gptel model name to bedrock-mantle model ID.

IDs can be added or replaced by calling
\(push (model-name . \"model-id\") gptel-bedrock-mantle--model-ids).")

(defvar gptel--bedrock-mantle-models
  ;; The bedrock-mantle endpoint exposes the OpenAI Responses API; the OpenAI
  ;; models below are the primary use case.  Mantle hosts other providers too
  ;; (Anthropic, DeepSeek, Qwen, Mistral Large 3, ...) -- add those symbols to
  ;; the backend's `:models' and register their IDs in
  ;; `gptel-bedrock-mantle--model-ids' as needed.
  ;; https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
  '((gpt-5.5
     :description "Best intelligence at scale for agentic, coding, and professional workflows"
     :capabilities (media tool-use json url)
     :mime-types ("image/jpeg" "image/png" "image/gif" "image/webp")
     :context-window 400)
    (gpt-5.4
     :description "Strong general-purpose model for coding and agentic tasks"
     :capabilities (media tool-use json url)
     :mime-types ("image/jpeg" "image/png" "image/gif" "image/webp")
     :context-window 400)
    (gpt-oss-120b
     :description "Open-weight reasoning model (120B)"
     :capabilities (tool-use json reasoning)
     :context-window 128)
    (gpt-oss-20b
     :description "Open-weight reasoning model (20B)"
     :capabilities (tool-use json reasoning)
     :context-window 128)
    (gpt-oss-safeguard-120b
     :description "Open-weight safety classification model (120B)"
     :capabilities (tool-use json)
     :context-window 128)
    (gpt-oss-safeguard-20b
     :description "Open-weight safety classification model (20B)"
     :capabilities (tool-use json)
     :context-window 128))
  "List of OpenAI models available on the AWS Bedrock bedrock-mantle endpoint.

See `gptel--openai-models' for the meaning of the per-model property
keys.")

(defun gptel-bedrock-mantle--get-model-id (backend model)
  "Return the bedrock-mantle model ID for MODEL on BACKEND.

OpenAI (and other mantle-only) models are resolved via
`gptel-bedrock-mantle--model-ids'.  Any other model falls back to the
shared `gptel-bedrock--get-model-id', which applies the backend's
`model-region' cross-region inference prefix."
  (or (alist-get model gptel-bedrock-mantle--model-ids nil nil #'eq)
      (gptel-bedrock--get-model-id
       model (gptel-bedrock-mantle-model-region backend))))

(cl-defmethod gptel--request-data ((backend gptel-bedrock-mantle) _prompts)
  "Prepare request data for the bedrock-mantle BACKEND.

This reuses the OpenAI Responses API format (built by the parent
method) but substitutes the Bedrock model ID for the `:model' field."
  (let ((data (cl-call-next-method)))
    (plist-put data :model
               (gptel-bedrock-mantle--get-model-id backend gptel-model))))

(defun gptel-bedrock-mantle--curl-args (region profile)
  "Generate curl arguments to sign a bedrock-mantle request for REGION.

PROFILE specifies the AWS profile to use for `aws configure
export-credentials'.  These arguments are only used for SigV4
authentication; bearer-token authentication is handled via the
backend header."
  (cl-multiple-value-bind (key-id secret token)
      (gptel-bedrock--get-credentials profile)
    (append
     (list "--user" (format "%s:%s" key-id secret)
           "--aws-sigv4" (format "aws:amz:%s:bedrock" region))
     (when token (list "-H" (format "x-amz-security-token: %s" token))))))

;;;###autoload
(cl-defun gptel-make-bedrock-mantle
    (name &key
          region
          (models gptel--bedrock-mantle-models)
          (model-region nil)
          stream curl-args request-params
          aws-profile aws-bearer-token
          (protocol "https"))
  "Register an AWS Bedrock (bedrock-mantle) backend for gptel with NAME.

The bedrock-mantle endpoint speaks the OpenAI-compatible Responses
API.

Keyword arguments:

REGION - AWS region name (e.g. \"us-east-2\")
MODELS - The list of models supported by this backend
MODEL-REGION - one of apac, eu, us, global or nil.  Used to build a
cross-region inference profile ID for models that require one.  The
OpenAI models on mantle do not support cross-region inference, so this
is ignored for them.
AWS-PROFILE - the aws profile to use for `aws configure export-credentials'
AWS-BEARER-TOKEN - a Bedrock API key (bearer token) for authentication
CURL-ARGS - additional curl args
STREAM - Whether to use streaming responses or not.
REQUEST-PARAMS - a plist of additional HTTP request
parameters (as plist keys) and values supported by the API.

Authentication precedence mirrors `gptel-make-bedrock':
- AWS-BEARER-TOKEN argument
- AWS_BEARER_TOKEN_BEDROCK environment variable
- AWS-PROFILE argument
- AWS_PROFILE environment variable
- AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN

When a bearer token is used, authentication is a simple Authorization
header and works without curl.  When SigV4 signing is used (profile or
IAM credentials), gptel requires curl >= 8.9 to compute the signature."
  (declare (indent 1))
  (unless (or aws-bearer-token (getenv "AWS_BEARER_TOKEN_BEDROCK"))
    (unless (and gptel-use-curl (version<= "8.9" (gptel-bedrock--curl-version)))
      (error "Bedrock-mantle SigV4 auth requires curl >= 8.9, but gptel-use-curl := %s, curl-version := %s"
             gptel-use-curl (gptel-bedrock--curl-version))))
  (let* ((host (format "bedrock-mantle.%s.api.aws" region))
         (endpoint "/openai/v1/responses"))
    (setf (alist-get name gptel--known-backends nil nil #'equal)
          (gptel--make-bedrock-mantle
           :name name
           :host host
           :header
           (lambda (_info)
             (when-let* ((token (or aws-bearer-token
                                    (getenv "AWS_BEARER_TOKEN_BEDROCK"))))
               `(("Authorization" . ,(concat "Bearer " token)))))
           :models (gptel--process-models models)
           :model-region model-region
           :protocol protocol
           :endpoint endpoint
           :stream stream
           :curl-args
           (lambda ()
             (append curl-args
                     (unless (or aws-bearer-token
                                 (getenv "AWS_BEARER_TOKEN_BEDROCK"))
                       (gptel-bedrock-mantle--curl-args region aws-profile))))
           :request-params request-params
           :url (concat protocol "://" host endpoint)))))

(provide 'gptel-bedrock-mantle)
;;; gptel-bedrock-mantle.el ends here
