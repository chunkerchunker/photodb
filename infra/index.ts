import * as fs from "node:fs";
import * as aws from "@pulumi/aws";
import * as pulumi from "@pulumi/pulumi";
import * as tls from "@pulumi/tls";

const config = new pulumi.Config();
const allowedIps = config.requireObject<string[]>("allowedIps");
const allowedCidrs = allowedIps.map((ip) => `${ip}/32`);

// --- SSH Key Pair ---

const sshKey = new tls.PrivateKey("photodb", {
  algorithm: "ED25519",
});

const keyPair = new aws.ec2.KeyPair("photodb", {
  keyName: "photodb",
  publicKey: sshKey.publicKeyOpenssh,
});

// --- Security Group ---

const sg = new aws.ec2.SecurityGroup("photodb", {
  description: "PhotoDB web app",
  ingress: [
    {
      description: "SSH",
      protocol: "tcp",
      fromPort: 22,
      toPort: 22,
      cidrBlocks: allowedCidrs,
    },
    {
      description: "Web",
      protocol: "tcp",
      fromPort: 3000,
      toPort: 3000,
      cidrBlocks: allowedCidrs,
    },
  ],
  egress: [
    {
      protocol: "-1",
      fromPort: 0,
      toPort: 0,
      cidrBlocks: ["0.0.0.0/0"],
    },
  ],
  tags: { Name: "photodb" },
});

// --- AMI (Amazon Linux 2023 ARM) ---

const ami = aws.ec2.getAmiOutput({
  mostRecent: true,
  owners: ["amazon"],
  filters: [
    { name: "name", values: ["al2023-ami-*-arm64"] },
    { name: "architecture", values: ["arm64"] },
    { name: "state", values: ["available"] },
  ],
});

// --- EC2 Instance ---

const userData = fs.readFileSync("user-data.sh", "utf-8");

const instance = new aws.ec2.Instance("photodb", {
  instanceType: "t4g.small",
  ami: ami.id,
  keyName: keyPair.keyName,
  vpcSecurityGroupIds: [sg.id],
  rootBlockDevice: {
    volumeSize: 40,
    volumeType: "gp3",
    deleteOnTermination: false,
  },
  userData,
  tags: { Name: "photodb" },
});

// --- Elastic IP (stable address across stop/start) ---

const eip = new aws.ec2.Eip("photodb", {
  instance: instance.id,
  tags: { Name: "photodb" },
});

// --- Outputs ---

export const publicIp = eip.publicIp;
export const instanceId = instance.id;
export const sshPrivateKey = pulumi.secret(sshKey.privateKeyOpenssh);
export const sshCommand = pulumi.interpolate`ssh -i ~/.ssh/photodb ec2-user@${eip.publicIp}`;
