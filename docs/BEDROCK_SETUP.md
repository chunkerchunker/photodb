# Using Amazon Bedrock with PhotoDB

PhotoDB supports using Anthropic Claude models via Amazon Bedrock as an alternative to the direct Anthropic API through the unified `instructor` library interface. This provides a seamless way to switch between providers without code changes.

## Prerequisites

1. **AWS Account**: You need an active AWS account with Bedrock access
2. **Model Access**: Request access to Claude models in Bedrock:
   - Go to AWS Console → Amazon Bedrock → Model access
   - Request access to Anthropic Claude models (particularly Claude 3.5 Sonnet)
   - Wait for approval (usually instant for Claude models)
3. **AWS Credentials**: Configure AWS credentials using one of:
   - AWS CLI: `aws configure`
   - Environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
   - IAM role (if running on EC2)
   - AWS profile

## Configuration

### Environment Variables

Set these environment variables to use Bedrock:

```bash
# Required: Switch to Bedrock provider
export LLM_PROVIDER=bedrock

# Optional: Specify model ID (defaults to Claude 3.5 Sonnet v2)
export BEDROCK_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"

# Optional: AWS configuration
export AWS_REGION=us-east-1  # Default: us-east-1
export AWS_PROFILE=myprofile  # Optional: use specific AWS profile

# Required for batch processing: S3 bucket for batch input/output
export BEDROCK_BATCH_S3_BUCKET=your-batch-processing-bucket

# Required for batch processing: IAM role ARN that Bedrock can assume
export BEDROCK_BATCH_ROLE_ARN=arn:aws:iam::123456789012:role/BedrockBatchRole
```

### Available Models

Bedrock model IDs for Claude:
- `anthropic.claude-3-5-sonnet-20241022-v2:0` - Claude 3.5 Sonnet v2 (recommended)
- `anthropic.claude-3-5-sonnet-20240620-v1:0` - Claude 3.5 Sonnet v1
- `anthropic.claude-3-haiku-20240307-v1:0` - Claude 3 Haiku (faster, cheaper)
- `anthropic.claude-3-opus-20240229-v1:0` - Claude 3 Opus (most capable)

### Usage Example

```bash
# Set up environment
export LLM_PROVIDER=bedrock
export AWS_REGION=us-west-2
export BEDROCK_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"

# Run photo processing with Bedrock
uv run process-photos /path/to/photos --stage enrich --parallel 10
```

## AWS IAM Permissions

Your AWS user/role needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
      ]
    }
  ]
}
```

## Cost Comparison

### Bedrock Pricing (as of 2024)
- **Claude 3.5 Sonnet**: $3.00 per million input tokens, $15.00 per million output tokens
- **Claude 3 Haiku**: $0.25 per million input tokens, $1.25 per million output tokens
- **Claude 3 Opus**: $15.00 per million input tokens, $75.00 per million output tokens

### Direct Anthropic API Pricing
- Same base prices as Bedrock
- **Batch API**: 50% discount (not available in Bedrock)

## Batch Processing

Bedrock supports batch processing with significant cost savings (50% discount) compared to real-time inference:

### Setup for Batch Processing

1. **S3 Bucket**: Create an S3 bucket for batch input/output files

2. **IAM Role for Bedrock**: Create an IAM role that Bedrock can assume to access your S3 bucket:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**Role Policy** (attach to the role):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-batch-processing-bucket",
        "arn:aws:s3:::your-batch-processing-bucket/*"
      ]
    }
  ]
}
```

3. **Your IAM Permissions**: Ensure your AWS credentials have access to:
   - `bedrock:CreateModelInvocationJob`
   - `bedrock:GetModelInvocationJob` 
   - `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` on your batch bucket
   - `iam:PassRole` to pass the Bedrock role

```bash
# Enable batch processing
export LLM_PROVIDER=bedrock
export BEDROCK_BATCH_S3_BUCKET=your-batch-processing-bucket
export BEDROCK_BATCH_ROLE_ARN=arn:aws:iam::123456789012:role/BedrockBatchRole

# Use batch mode
uv run process-photos /path/to/photos --stage enrich --batch-mode --batch-size 100
```

## Limitations

When using Bedrock instead of the direct Anthropic API:

1. **Batch Setup**: Requires S3 bucket configuration for batch processing
2. **Rate Limits**: Subject to AWS Bedrock rate limits (varies by region and model)
3. **Latency**: May have slightly higher latency due to AWS infrastructure
4. **Region Availability**: Not all AWS regions have Bedrock or all Claude models

## Switching Between Providers

You can easily switch between Anthropic API and Bedrock:

```bash
# Use direct Anthropic API
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-api-key
uv run process-photos /path/to/photos --stage enrich

# Use Bedrock
export LLM_PROVIDER=bedrock
# (AWS credentials configured)
uv run process-photos /path/to/photos --stage enrich
```

## Troubleshooting

### Access Denied Error
```
Error: User is not authorized to perform: bedrock:InvokeModel
```
**Solution**: Ensure your AWS user/role has the required IAM permissions and you've requested model access in the Bedrock console.

### Model Not Found
```
Error: Model anthropic.claude-3-5-sonnet not found
```
**Solution**: Use the full model ID with version, e.g., `anthropic.claude-3-5-sonnet-20241022-v2:0`

### Region Not Supported
```
Error: Bedrock is not available in region eu-central-1
```
**Solution**: Use a supported region like `us-east-1` or `us-west-2`. Check AWS documentation for current region availability.

### Slow Processing
If processing is slower than expected:
1. Consider using Claude 3 Haiku for faster, cheaper processing
2. Increase parallelism: `--parallel 50`
3. Check AWS CloudWatch for throttling metrics
4. Consider using the direct Anthropic API with batch mode for bulk processing

## Monitoring

Monitor your Bedrock usage in AWS:
- **CloudWatch Metrics**: View invocation counts, latency, and errors
- **AWS Cost Explorer**: Track spending on Bedrock
- **CloudTrail**: Audit API calls for compliance

## Best Practices

1. **Use Haiku for Testing**: Start with Claude 3 Haiku for development to reduce costs
2. **Set Quotas**: Configure AWS budget alerts to avoid unexpected charges
3. **Cache Results**: PhotoDB already caches results in PostgreSQL - avoid reprocessing
4. **Regional Deployment**: Deploy in the same AWS region as your Bedrock endpoint to reduce latency
5. **Error Handling**: The Bedrock implementation includes retry logic for transient errors