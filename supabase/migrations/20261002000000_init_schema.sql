-- VietReceipt IDP: schema khởi tạo (documents, receipts, receipt_items, processing_runs, validations, reviews)
-- Tiền tệ V1 là VND. Cột tiền dùng numeric để export-ready cho DWH/ERP.
-- RLS bật trên mọi bảng và KHÔNG có policy: chỉ service role (backend) truy cập được; anon/authenticated bị chặn.

-- ============ Enums ============
create type public.document_status as enum
  ('uploaded', 'queued', 'processing', 'completed', 'needs_review', 'failed');

create type public.routing_decision as enum
  ('auto_accept', 'accept_with_warning', 'human_review');

create type public.run_status as enum
  ('running', 'succeeded', 'failed');

create type public.payment_method as enum
  ('cash', 'card', 'ewallet', 'bank_transfer', 'other');

create type public.validation_severity as enum
  ('error', 'warning');

create type public.review_status as enum
  ('pending', 'confirmed', 'corrected');

-- ============ Helper ============
create or replace function public.set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

-- ============ documents: file gốc (ảnh/PDF) ============
create table public.documents (
  id                uuid primary key default gen_random_uuid(),
  storage_bucket    text not null default 'receipts-original',
  storage_path      text not null,
  original_filename text,
  mime_type         text,
  file_size_bytes   bigint check (file_size_bytes is null or file_size_bytes >= 0),
  sha256            text,
  page_count        integer not null default 1 check (page_count >= 1),
  status            public.document_status not null default 'uploaded',
  created_at        timestamptz not null default now(),
  updated_at        timestamptz not null default now(),
  unique (storage_bucket, storage_path)
);
create unique index documents_sha256_key on public.documents (sha256) where sha256 is not null;
create index documents_status_idx on public.documents (status);
create trigger documents_set_updated_at before update on public.documents
  for each row execute function public.set_updated_at();

-- ============ processing_runs: mỗi lần chạy pipeline ============
create table public.processing_runs (
  id            uuid primary key default gen_random_uuid(),
  document_id   uuid not null references public.documents(id) on delete cascade,
  attempt       integer not null default 1 check (attempt >= 1),
  status        public.run_status not null default 'running',
  model_version text,          -- vd "ocr=paddle-v5+layoutxlm-ft-2026-11-20"
  config        jsonb,         -- preprocessing/OCR/KIE config đã dùng
  raw_json      jsonb,         -- token, bbox, ocr_conf, kie_prob (để tune ngưỡng ở M5) + audit metadata
  mlflow_run_id text,
  latency_ms    integer check (latency_ms is null or latency_ms >= 0),
  error         text,
  started_at    timestamptz not null default now(),
  finished_at   timestamptz,
  created_at    timestamptz not null default now(),
  unique (document_id, attempt)
);
create index processing_runs_document_idx on public.processing_runs (document_id);

-- ============ receipts: kết quả trích xuất (1 receipt / document trong V1) ============
create table public.receipts (
  id                 uuid primary key default gen_random_uuid(),
  document_id        uuid not null unique references public.documents(id) on delete cascade,
  run_id             uuid references public.processing_runs(id) on delete set null,
  -- merchant / giao dịch (nullable: tài liệu cần review có thể thiếu field; tính bắt buộc kiểm ở validation)
  merchant_name      text,
  merchant_address   text,
  tax_code           text,      -- text để không mất leading zero
  invoice_id         text,
  transaction_date   date,
  transaction_time   time,
  payment_method     public.payment_method,
  currency           text not null default 'VND',
  price_includes_vat boolean,
  -- tiền (VND)
  subtotal           numeric(18,2) check (subtotal is null or subtotal >= 0),
  vat_amount         numeric(18,2) check (vat_amount is null or vat_amount >= 0),
  discount_amount    numeric(18,2) check (discount_amount is null or discount_amount >= 0),
  service_charge     numeric(18,2) check (service_charge is null or service_charge >= 0),
  total_amount       numeric(18,2) check (total_amount is null or total_amount >= 0),
  -- confidence & routing
  doc_confidence     numeric(5,4) check (doc_confidence is null or doc_confidence between 0 and 1),
  routing            public.routing_decision,
  source             text not null default 'model' check (source in ('model', 'human')),  -- giá trị hiện tại do model hay đã sửa tay
  model_version      text,
  raw_json           jsonb,     -- payload API đầy đủ (kể cả field chưa map cột)
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now()
);
create index receipts_merchant_idx on public.receipts (merchant_name);
create index receipts_date_idx on public.receipts (transaction_date);
create index receipts_routing_idx on public.receipts (routing);
create trigger receipts_set_updated_at before update on public.receipts
  for each row execute function public.set_updated_at();

-- ============ receipt_items: line items ============
create table public.receipt_items (
  id              uuid primary key default gen_random_uuid(),
  receipt_id      uuid not null references public.receipts(id) on delete cascade,
  line_no         integer not null check (line_no >= 1),
  raw_name        text,         -- text OCR gốc
  name            text,
  normalized_name text,
  quantity        numeric(12,3) check (quantity is null or quantity >= 0),   -- cho phép số lẻ theo cân (0,352 kg)
  unit_price      numeric(18,2) check (unit_price is null or unit_price >= 0),
  amount          numeric(18,2) check (amount is null or amount >= 0),
  confidence      numeric(5,4) check (confidence is null or confidence between 0 and 1),
  created_at      timestamptz not null default now(),
  unique (receipt_id, line_no)
);

-- ============ validations: kết quả từng rule ============
create table public.validations (
  id            uuid primary key default gen_random_uuid(),
  document_id   uuid not null references public.documents(id) on delete cascade,
  run_id        uuid references public.processing_runs(id) on delete cascade,
  receipt_id    uuid references public.receipts(id) on delete cascade,
  item_id       uuid references public.receipt_items(id) on delete cascade,
  rule_name     text not null,      -- vd "item_amount", "total_sum", "date_parse"
  rule_matched  text,               -- công thức khớp: "vat_exclusive" | "vat_inclusive"
  passed        boolean not null,
  severity      public.validation_severity not null default 'error',
  expected      numeric(18,2),
  actual        numeric(18,2),
  message       text,
  created_at    timestamptz not null default now()
);
create index validations_document_idx on public.validations (document_id);
create index validations_failed_idx on public.validations (document_id) where not passed;

-- ============ reviews: hàng đợi review + correction store ============
create table public.reviews (
  id             uuid primary key default gen_random_uuid(),
  document_id    uuid not null references public.documents(id) on delete cascade,
  receipt_id     uuid references public.receipts(id) on delete set null,
  status         public.review_status not null default 'pending',
  predicted_json jsonb not null,
  corrected_json jsonb,
  reviewer       text,
  reason         text,
  reviewed_at    timestamptz,
  exported_at    timestamptz,   -- đã export sang annotation/retraining dataset chưa
  created_at     timestamptz not null default now(),
  check (status = 'pending' or (reviewer is not null and reviewed_at is not null)),
  check (status <> 'corrected' or corrected_json is not null)
);
create index reviews_queue_idx on public.reviews (created_at) where status = 'pending';
create index reviews_export_idx on public.reviews (reviewed_at) where exported_at is null and status = 'corrected';

-- ============ RLS: chặn truy cập qua API công khai ============
alter table public.documents       enable row level security;
alter table public.processing_runs enable row level security;
alter table public.receipts        enable row level security;
alter table public.receipt_items   enable row level security;
alter table public.validations     enable row level security;
alter table public.reviews         enable row level security;

-- ============ Storage: bucket private cho ảnh/PDF gốc ============
insert into storage.buckets (id, name, public)
values ('receipts-original', 'receipts-original', false)
on conflict (id) do nothing;
