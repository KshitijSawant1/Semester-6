Good decision. We will perform **Experiment 7 using Terraform locally (no AWS required)** step by step in a clean, exam-ready way.

---

# Step-by-Step Execution (Local Terraform Experiment)

## Step 1: Install Terraform

Download Terraform from official website and install.

### Verify Installation

```bash
terraform version
```

Expected: Terraform version should display.

---

## Step 2: Create Project Folder

```bash
mkdir terraform_project
cd terraform_project
```

---

## Step 3: Create Terraform File

Create a file named:

```bash
touch main.tf
```

Open `main.tf` and add the following code:

```hcl
terraform {
  required_providers {
    local = {
      source  = "hashicorp/local"
      version = "2.4.0"
    }
  }
}

provider "local" {}

resource "local_file" "example" {
  filename = "demo.txt"
  content  = "Hello, Terraform Local Infrastructure!"
}
```

---

## Step 4: Initialize Terraform

```bash
terraform init
```

What happens:

- Downloads provider
- Prepares environment

---

## Step 5: Validate Configuration (Optional but recommended)

```bash
terraform validate
```

Expected:

```
Success! The configuration is valid.
```

---

## Step 6: Preview Infrastructure

```bash
terraform plan
```

What you will see:

- Terraform shows what will be created
- A file `demo.txt` will be added

---

## Step 7: Apply Configuration

```bash
terraform apply
```

Then type:

```bash
yes
```

---

## Step 8: Verify Output

Check your folder:

```bash
ls
```

You should see:

```
demo.txt
```

Open it:

```bash
cat demo.txt
```

Expected:

```
Hello, Terraform Local Infrastructure!
```

---

## Step 9: Modify Infrastructure

Edit `main.tf`:

```hcl
content = "Updated Terraform File"
```

Then run:

```bash
terraform apply
```

✔ File content will update automatically

---

## Step 10: Destroy Infrastructure

```bash
terraform destroy
```

Type:

```bash
yes
```

✔ File will be deleted

---

# Final Understanding (Important for Viva)

Terraform lifecycle:

1. init → setup environment
2. validate → check syntax
3. plan → preview changes
4. apply → create infrastructure
5. destroy → remove infrastructure

---

# What You Achieved

- Used Terraform without cloud
- Created infrastructure (file) using IaC
- Managed lifecycle (create → update → delete)

---

# If Examiner Asks

**Q: Why local provider instead of AWS?**

Answer:

Terraform concepts can be demonstrated locally without cloud dependency. The same workflow applies to real cloud infrastructure with provider changes.

---

# If You Want Next

I can:

- Convert this into **perfect lab record (Aim, Theory, Result)**
- Give **implementation table (Steps | Title | Command)**
- Provide **viva Q&A**

Just tell me 👍
